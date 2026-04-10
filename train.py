"""
Deep Metric Learning — Oxford-IIIT Pet Dataset
Optimized for NVIDIA H100 (BF16, TF32, large batches, high worker count)
"""

import itertools
import os
import random
import re
import time

import kagglehub
import matplotlib
matplotlib.use("Agg")  # no display on headless server
import matplotlib.pyplot as plt
import numpy as np
import umap
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision  # noqa: F401

from pytorch_metric_learning import losses, miners, samplers


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS        = 30
BATCH_SIZE    = 256     # H100 has 80 GB VRAM; 256 is comfortable with BF16
EMBED_DIM     = 128
NUM_WORKERS   = 16      # match to your server's CPU core count
M_PER_CLASS   = 8       # samples per class per mini-batch (MPerClassSampler)
LR_BACKBONE   = 1e-4
LR_HEAD       = 1e-3
MARGIN        = 0.2     # triplet loss margin
SEED          = 42
OUT_DIR       = "."     # root for /models and /img output folders

# Breeds withheld entirely for few-shot evaluation (never seen during training)
HELD_OUT_BREEDS = [
    "american_bulldog",
    "american_pit_bull_terrier",  # visually similar to american_bulldog
    "saint_bernard",
    "Bengal",
    "Siamese",
]

# Breeds highlighted in the zoomed UMAP plot
FOCUS_BREEDS = [
    "yorkshire_terrier",
    "scottish_terrier",
    "wheaten_terrier",
    "staffordshire_bull_terrier",
    "miniature_pinscher",
]

# ── H100-specific global flags ────────────────────────────────────────────────
# TF32 gives near-FP32 accuracy at ~3× the throughput on Ampere/Hopper
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # auto-tune conv kernels for fixed input shapes


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_breed(filename: str):
    name  = filename.rsplit(".", 1)[0]
    breed = re.sub(r"_\d+$", "", name)
    species = "cat" if breed[0].isupper() else "dog"
    return species, breed


def build_class_to_idx(imgdir: str):
    breeds = set()
    for fn in os.listdir(imgdir):
        if fn.endswith(".jpg"):
            _, breed = get_breed(fn)
            breeds.add(breed)
    breed_list   = sorted(breeds)
    class_to_idx = {b: i for i, b in enumerate(breed_list)}
    species_to_idx = {"cat": 0, "dog": 1}
    return class_to_idx, species_to_idx


class PetDataset(Dataset):
    def __init__(self, imgdir, classes, speciesdict, transform=None):
        self.imgdir    = imgdir
        self.classes   = classes
        self.species   = speciesdict
        self.transform = transform
        self.samples   = []
        self.breeddict = {b: [] for b in classes}

        for idx, fn in enumerate(os.listdir(imgdir)):
            if fn.endswith(".jpg"):
                species, breed = get_breed(fn)
                self.samples.append((fn, classes[breed], speciesdict[species]))
                self.breeddict[breed].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn, label, species = self.samples[idx]
        img = Image.open(os.path.join(self.imgdir, fn)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, species


class TransformSubset(Dataset):
    """Apply a transform to a Subset without duplicating data."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label, species = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, species


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        backbone    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)
        emb  = self.projector(feat)
        return nn.functional.normalize(emb, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_pairs(labels, n=5000, seed=42):
    rng = random.Random(seed)
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(l, []).append(i)

    pos = []
    for idxs in label_to_idx.values():
        pos.extend(itertools.combinations(idxs, 2))
    pos = rng.sample(pos, min(n, len(pos)))

    all_labs = list(label_to_idx.keys())
    neg = []
    while len(neg) < n:
        l1, l2 = rng.sample(all_labs, 2)
        neg.append((rng.choice(label_to_idx[l1]), rng.choice(label_to_idx[l2])))
    return pos, neg


def precision_at_k(query_label, retrieved_labels, k):
    return np.mean(retrieved_labels[:k] == query_label)


def recall_at_k(query_label, retrieved_labels, all_labels, k):
    total = np.sum(all_labels == query_label) - 1
    found = np.sum(retrieved_labels[:k] == query_label)
    return found / max(total, 1)


def run_episodes(embeddings, labels, n_way=5, k_shot=1, n_episodes=100, n_query=15, seed=42):
    rng = np.random.default_rng(seed)
    accs = []
    unique_classes = np.unique(labels)

    for _ in range(n_episodes):
        chosen = rng.choice(unique_classes, n_way, replace=False)
        support_embs, query_embs, query_y = [], [], []

        valid = True
        for i, c in enumerate(chosen):
            idxs = np.where(labels == c)[0]
            if len(idxs) < k_shot + n_query:
                valid = False
                break
            pick = rng.choice(idxs, k_shot + n_query, replace=False)
            support_embs.append(embeddings[pick[:k_shot]])
            query_embs.append(embeddings[pick[k_shot:]])
            query_y += [i] * n_query

        if not valid:
            continue

        protos = np.array([s.mean(axis=0) for s in support_embs])
        norms  = np.linalg.norm(protos, axis=1, keepdims=True)
        protos = protos / np.where(norms == 0, 1, norms)
        sims   = np.vstack(query_embs) @ protos.T
        preds  = np.argmax(sims, axis=1)
        accs.append(np.mean(preds == np.array(query_y)))

    return float(np.mean(accs)), float(np.std(accs))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (all use savefig — no show() on a headless server)
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(train_set, val_set, test_set, data, classes, classes_inv, out):
    def counts(subset):
        c = {b: 0 for b in classes}
        for idx in subset.indices:
            c[classes_inv[data.samples[idx][1]]] += 1
        return c

    tc, vc, sc = counts(train_set), counts(val_set), counts(test_set)
    breeds = list(tc.keys())
    x, w   = np.arange(len(breeds)), 0.25

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x - w, [tc[b] for b in breeds], w, label="Train")
    ax.bar(x,     [vc[b] for b in breeds], w, label="Val")
    ax.bar(x + w, [sc[b] for b in breeds], w, label="Test")
    ax.set_xticks(x); ax.set_xticklabels(breeds, rotation=45, ha="right")
    ax.set_xlabel("Breed"); ax.set_ylabel("Count")
    ax.set_title("Class Distribution"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "img", "class_distributions.png"))
    plt.close()
    print("  Saved class_distributions.png")


def plot_loss_curves(train_losses, val_losses, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses,   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "img", "loss_curves.png"))
    plt.close()
    print("  Saved loss_curves.png")


def plot_roc(fpr, tpr, eer, eer_idx, auc, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, color="#c4622a")
    ax.plot([0, 1], [0, 1], "--", color="#ccc")
    ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="#d44", s=60, zorder=5,
               label=f"EER = {eer:.3f}")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Verification ROC (AUC = {auc:.3f})"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "img", "roc_curve.png"), dpi=150)
    plt.close()
    print("  Saved roc_curve.png")


def plot_few_shot(acc1, std1, acc5, std5, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels_bar = ["Random\nBaseline", "1-Shot", "5-Shot"]
    values = [0.20, acc1, acc5]
    errors = [0.00, std1, std5]
    colors = ["#cccccc", "#3d7ec7", "#c4622a"]
    bars = ax.bar(labels_bar, values, yerr=errors, capsize=6, color=colors,
                  edgecolor="white", linewidth=1.2,
                  error_kw=dict(ecolor="#555", lw=1.5))
    for bar, v, e in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.20, color="#999", linestyle="--", linewidth=1, label="Random (20%)")
    ax.set_ylim(0, min(1.05, max(values) + 0.15))
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("5-Way Few-Shot Classification Accuracy\n(100 episodes, held-out breeds)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "img", "few_shot_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved few_shot_accuracy.png")


def plot_umap(emb_2d, labels, classes_inv, sil, focus_breeds, classes, out):
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    # Full scatter
    fig, ax = plt.subplots(figsize=(14, 11))
    for i, l in enumerate(unique_labels):
        mask = labels == l
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[cmap(i % 20)],
                   s=12, alpha=0.75, label=classes_inv[l])
    ax.legend(fontsize=6, ncol=3, loc="center left", bbox_to_anchor=(1.0, 0.5),
              markerscale=2.5, frameon=True, framealpha=0.8)
    ax.set_title(f"Embedding Space — UMAP Projection\n(Silhouette: {sil:.4f})",
                 fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out, "img", "umap_full.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved umap_full.png")

    # Zoomed view (similar terrier breeds)
    valid = [b for b in focus_breeds if b in classes and classes[b] in unique_labels]
    if len(valid) >= 2:
        focus_ids   = [classes[b] for b in valid]
        cmap_focus  = plt.cm.get_cmap("Set1", len(valid))
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (b, fid) in enumerate(zip(valid, focus_ids)):
            m = labels == fid
            ax.scatter(emb_2d[m, 0], emb_2d[m, 1], c=[cmap_focus(i)],
                       s=25, alpha=0.85, label=b)
        ax.legend(fontsize=9, markerscale=1.8)
        ax.set_title("Zoomed View — Visually Similar Terrier Breeds",
                     fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(out, "img", "umap_zoomed.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print("  Saved umap_zoomed.png")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction (shared utility)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    all_embs, all_labels = [], []
    for imgs, lbls, _ in loader:
        embs = model(imgs.to(device, non_blocking=True)).cpu().numpy()
        all_embs.append(embs)
        all_labels.extend(lbls.numpy())
    return np.vstack(all_embs), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "img"),    exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset …")
    imgdir = (kagglehub.dataset_download(
                  "tanlikesmath/the-oxfordiiit-pet-dataset",
                  output_dir="./data")
              + "/images/")

    classes, speciesdict = build_class_to_idx(imgdir)
    classes_inv = {v: k for k, v in classes.items()}
    data = PetDataset(imgdir, classes, speciesdict)
    print(f"  Total images: {len(data)}, breeds: {len(classes)}")

    held_out_ids = [idx for b in HELD_OUT_BREEDS for idx in data.breeddict[b]]
    held_out_set = torch.utils.data.Subset(data, held_out_ids)

    remaining_ids    = [i for i in range(len(data)) if i not in set(held_out_ids)]
    remaining_labels = [data.samples[i][1] for i in remaining_ids]

    train_idx, val_idx = train_test_split(
        remaining_ids, test_size=0.2, stratify=remaining_labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        val_idx, test_size=0.5,
        stratify=[data.samples[i][1] for i in val_idx],
        random_state=SEED,
    )
    train_set = torch.utils.data.Subset(data, train_idx)
    val_set   = torch.utils.data.Subset(data, val_idx)
    test_set  = torch.utils.data.Subset(data, test_idx)
    print(f"  Split → train: {len(train_set)}, val: {len(val_set)}, "
          f"test: {len(test_set)}, held-out: {len(held_out_set)}")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = TransformSubset(train_set, train_transform)
    val_ds   = TransformSubset(val_set,   val_transform)
    test_ds  = TransformSubset(test_set,  val_transform)
    train_labels = [data.samples[i][1] for i in train_set.indices]

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,          # async CPU→GPU transfers
        persistent_workers=True,  # keep workers alive between epochs
    )
    sampler      = samplers.MPerClassSampler(train_labels, m=M_PER_CLASS,
                                             batch_size=BATCH_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, **loader_kwargs)

    plot_class_distribution(train_set, val_set, test_set,
                            data, classes, classes_inv, OUT_DIR)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/6] Building model …")
    model = EmbeddingNet(embed_dim=EMBED_DIM).to(device)
    # model = torch.compile(model)  # disabled: no C compiler in container  # graph-level fusion — first epoch is slower

    loss_fn = losses.TripletMarginLoss(margin=MARGIN)
    miner   = miners.TripletMarginMiner(margin=MARGIN, type_of_triplets="semihard")

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(),   "lr": LR_BACKBONE},
        {"params": model.projector.parameters(), "lr": LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # BF16 is the preferred dtype on H100 — wider dynamic range than FP16,
    # no loss scaling needed, hardware-native on Hopper.
    # GradScaler is kept for FP16 fallback compatibility but is a near-no-op in BF16.
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    scaler = GradScaler(device=device, enabled=(dtype == torch.float16))

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[3/6] Training for {EPOCHS} epochs …")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        t0 = time.time()

        model.train()
        epoch_loss = 0.0
        for imgs, lbls, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)  # non_blocking works because pin_memory=True
            lbls = lbls.to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=dtype):
                embs       = model(imgs)
                hard_pairs = miner(embs, lbls)
                loss       = loss_fn(embs, lbls, hard_pairs)
            optimizer.zero_grad(set_to_none=True)  # set_to_none saves a memset vs zero
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=dtype):
                    embs       = model(imgs)
                    hard_pairs = miner(embs, lbls)
                    loss       = loss_fn(embs, lbls, hard_pairs)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:>3}/{EPOCHS}  "
              f"train={train_losses[-1]:.4f}  val={val_losses[-1]:.4f}  "
              f"({elapsed:.1f}s)")

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            ckpt = os.path.join(OUT_DIR, "models", "best_model.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"    ✓ Checkpoint saved (val={best_val_loss:.4f})")

    plot_loss_curves(train_losses, val_losses, OUT_DIR)

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n[4/6] Verification (ROC / EER) …")
    embeddings, labels = extract_embeddings(model, test_loader, device)
    np.save(os.path.join(OUT_DIR, "test_embeddings.npy"), embeddings)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"),     labels)

    pos_pairs, neg_pairs = generate_pairs(labels, seed=SEED)
    pair_labels  = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    similarities = np.array([np.dot(embeddings[a], embeddings[b])
                              for a, b in pos_pairs + neg_pairs])

    fpr, tpr, _ = roc_curve(pair_labels, similarities)
    auc         = roc_auc_score(pair_labels, similarities)
    fnr         = 1 - tpr
    eer_idx     = int(np.nanargmin(np.abs(fpr - fnr)))
    eer         = float(fpr[eer_idx])
    print(f"  AUC: {auc:.4f}  |  EER: {eer:.4f}")
    plot_roc(fpr, tpr, eer, eer_idx, auc, OUT_DIR)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    print("\n[5/6] Retrieval (P@k, R@k) …")
    nn_model = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn_model.fit(embeddings)
    _, nn_indices    = nn_model.kneighbors(embeddings)
    neighbor_indices = nn_indices[:, 1:]  # drop self (col 0)

    for k in [1, 5]:
        p = np.mean([precision_at_k(labels[i], labels[neighbor_indices[i]], k)
                     for i in range(len(labels))])
        r = np.mean([recall_at_k(labels[i], labels[neighbor_indices[i]], labels, k)
                     for i in range(len(labels))])
        print(f"  P@{k}: {p:.4f}  |  R@{k}: {r:.4f}")

    # ── Few-shot classification ───────────────────────────────────────────────
    print("\n[6/6] Few-shot classification …")
    held_loader = DataLoader(
        TransformSubset(held_out_set, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs,
    )
    ho_embs, ho_labels = extract_embeddings(model, held_loader, device)

    acc1, std1 = run_episodes(ho_embs, ho_labels, k_shot=1, seed=SEED)
    acc5, std5 = run_episodes(ho_embs, ho_labels, k_shot=5, seed=SEED)
    print(f"  1-shot: {acc1:.3f} ± {std1:.3f}")
    print(f"  5-shot: {acc5:.3f} ± {std5:.3f}")
    print(f"  Random baseline: {1/5:.3f}")
    plot_few_shot(acc1, std1, acc5, std5, OUT_DIR)

    # ── UMAP + silhouette ─────────────────────────────────────────────────────
    print("\nComputing UMAP projection …")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
    emb_2d  = reducer.fit_transform(embeddings)
    sil     = silhouette_score(embeddings, labels, metric="cosine")
    print(f"  Silhouette score: {sil:.4f}")
    plot_umap(emb_2d, labels, classes_inv, sil, FOCUS_BREEDS, classes, OUT_DIR)

    print("\nDone. All outputs written to:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()