"""
Deep Metric Learning — Oxford-IIIT Pet Dataset
Optimized for NVIDIA H100 (BF16, TF32, large batches, high worker count)
v2: loads pre-trained checkpoint, skips training
    fixes PetDataset.breeddict index bug
"""

import itertools
import os
import random
import re

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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from pytorch_metric_learning import losses, miners, samplers


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE  = 256
EMBED_DIM   = 128
NUM_WORKERS = 16
MARGIN      = 0.2
SEED        = 42
OUT_DIR     = "."
CHECKPOINT  = "models/best_model.pth"  # path to saved checkpoint

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


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
    class_to_idx  = {b: i for i, b in enumerate(sorted(breeds))}
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

        # FIX: use a separate sample_idx counter instead of the enumerate idx,
        # which counted all files (including non-.jpg) and could exceed
        # len(self.samples), causing IndexError when accessed via Subset.
        sample_idx = 0
        for fn in os.listdir(imgdir):
            if fn.endswith(".jpg"):
                species, breed = get_breed(fn)
                self.samples.append((fn, classes[breed], speciesdict[species]))
                self.breeddict[breed].append(sample_idx)
                sample_idx += 1

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
        backbone     = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
# Plotting
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

    valid = [b for b in focus_breeds if b in classes and classes[b] in unique_labels]
    if len(valid) >= 2:
        focus_ids  = [classes[b] for b in valid]
        cmap_focus = plt.cm.get_cmap("Set1", len(valid))
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

    os.makedirs(os.path.join(OUT_DIR, "img"), exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
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

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = TransformSubset(test_set, val_transform)

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, **loader_kwargs)

    plot_class_distribution(train_set, val_set, test_set,
                            data, classes, classes_inv, OUT_DIR)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2/5] Loading checkpoint from {CHECKPOINT} …")
    model = EmbeddingNet(embed_dim=EMBED_DIM).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    # strip 'torch.compile' _orig_mod prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print("  Model loaded.")

    # ── Verification ──────────────────────────────────────────────────────────
    print("\n[3/5] Verification (ROC / EER) …")
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
    print("\n[4/5] Retrieval (P@k, R@k) …")
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
    print("\n[5/5] Few-shot classification …")
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
