"""
Experiment 1: Quantifying the Permutation Invariance Problem
============================================================
Goal: Show that naively averaging two transformers trained from the same
      initialization on different data splits degrades performance.

Setup:
  - Dataset: CIFAR-10
  - Model: Small Vision Transformer (ViT-Tiny)
  - Two clients train from the SAME initial weights on DIFFERENT data splits
  - We measure: accuracy of each individual model, the naive average, and
    a random-permutation baseline to show worst-case misalignment

Run:
  pip install torch torchvision matplotlib
  python experiment1.py

  On a single GPU this takes ~15-20 minutes total.
  On CPU, expect ~1-2 hours.

Output:
  - experiment1_results.pdf   (bar chart comparison)
  - experiment1_results.json  (raw numbers for your records)
  - Console summary table
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import json
import os
import time

# ──────────────────────────────────────────────────────────
# 1. SMALL VISION TRANSFORMER
# ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dim."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)    # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block."""
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    """
    Tiny Vision Transformer for CIFAR-10.
    Default: 4 layers, 4 heads, embed_dim=128, patch_size=4
    ~1.2M parameters — trains in minutes on a GPU.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=128, depth=4, num_heads=4,
                 mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                           in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])  # CLS token only
        return self.head(x)


# ──────────────────────────────────────────────────────────
# 2. DATA LOADING WITH NON-IID SPLITS
# ──────────────────────────────────────────────────────────

def get_cifar10_splits(data_dir='./data', heterogeneity='moderate'):
    """
    Create two non-IID client splits of CIFAR-10.

    heterogeneity levels:
      'mild':     Random 50/50 split (nearly IID)
      'moderate': Dirichlet alpha=0.5 (uneven class proportions)
      'extreme':  Client A gets classes 0-4, Client B gets classes 5-9
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test)

    targets = np.array(trainset.targets)
    n = len(targets)

    if heterogeneity == 'mild':
        indices = np.random.permutation(n)
        split = n // 2
        client_a_idx = indices[:split]
        client_b_idx = indices[split:]

    elif heterogeneity == 'moderate':
        alpha = 0.5
        client_a_idx, client_b_idx = [], []
        for c in range(10):
            class_idx = np.where(targets == c)[0]
            np.random.shuffle(class_idx)
            props = np.random.dirichlet([alpha, alpha])
            split = int(len(class_idx) * props[0])
            client_a_idx.extend(class_idx[:split])
            client_b_idx.extend(class_idx[split:])
        client_a_idx = np.array(client_a_idx)
        client_b_idx = np.array(client_b_idx)

    elif heterogeneity == 'extreme':
        client_a_idx = np.where(targets < 5)[0]
        client_b_idx = np.where(targets >= 5)[0]

    else:
        raise ValueError(f"Unknown heterogeneity: {heterogeneity}")

    print(f"\n  Split: {heterogeneity}")
    print(f"  Client A: {len(client_a_idx)} samples | "
          f"Client B: {len(client_b_idx)} samples")
    for name, idx in [("A", client_a_idx), ("B", client_b_idx)]:
        dist = np.bincount(targets[idx], minlength=10)
        print(f"  Client {name} class distribution: {dist}")

    train_a = Subset(trainset, client_a_idx)
    train_b = Subset(trainset, client_b_idx)
    return train_a, train_b, testset


# ──────────────────────────────────────────────────────────
# 3. TRAINING & EVALUATION
# ──────────────────────────────────────────────────────────

def train_model(model, train_loader, device, epochs=15, lr=1e-3):
    """Train and return per-epoch losses."""
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        epoch_losses.append(avg_loss)
        scheduler.step()
        print(f"    Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Train Acc: {acc:.1f}%")
    return epoch_losses


def evaluate(model, test_loader, device):
    """Return (accuracy%, avg_loss) on test set."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total, total_loss / len(test_loader)


# ──────────────────────────────────────────────────────────
# 4. WEIGHT AVERAGING METHODS
# ──────────────────────────────────────────────────────────

def naive_average(model_a, model_b):
    """Element-wise weight averaging (standard FedAvg)."""
    averaged = copy.deepcopy(model_a)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    avg_state = {k: (state_a[k] + state_b[k]) / 2.0 for k in state_a}
    averaged.load_state_dict(avg_state)
    return averaged


def permute_model(model, seed=42):
    """
    Randomly permute attention heads and FFN neurons in each layer.
    The permuted model computes the SAME function (sanity check:
    its accuracy should be unchanged), but its raw weights are
    reordered — simulating maximum misalignment between clients.
    """
    rng = np.random.RandomState(seed)
    permuted = copy.deepcopy(model)
    state = permuted.state_dict()

    for block_idx in range(len(model.blocks)):
        pfx = f"blocks.{block_idx}"
        embed_dim = model.blocks[block_idx].attn.embed_dim
        num_heads = model.blocks[block_idx].attn.num_heads
        head_dim = embed_dim // num_heads

        # ── Permute attention heads ──
        head_perm = rng.permutation(num_heads)

        in_w = state[f"{pfx}.attn.in_proj_weight"].clone()
        in_b = state[f"{pfx}.attn.in_proj_bias"].clone()
        out_w = state[f"{pfx}.attn.out_proj.weight"].clone()

        new_in_w, new_in_b = in_w.clone(), in_b.clone()
        new_out_w = out_w.clone()

        for qkv in range(3):  # Q, K, V blocks
            offset = qkv * embed_dim
            for new_h, old_h in enumerate(head_perm):
                src = slice(offset + old_h * head_dim,
                            offset + old_h * head_dim + head_dim)
                dst = slice(offset + new_h * head_dim,
                            offset + new_h * head_dim + head_dim)
                new_in_w[dst] = in_w[src]
                new_in_b[dst] = in_b[src]

        for new_h, old_h in enumerate(head_perm):
            src = slice(old_h * head_dim, old_h * head_dim + head_dim)
            dst = slice(new_h * head_dim, new_h * head_dim + head_dim)
            new_out_w[:, dst] = out_w[:, src]

        state[f"{pfx}.attn.in_proj_weight"] = new_in_w
        state[f"{pfx}.attn.in_proj_bias"] = new_in_b
        state[f"{pfx}.attn.out_proj.weight"] = new_out_w

        # ── Permute FFN intermediate neurons ──
        mlp_dim = state[f"{pfx}.mlp.0.weight"].shape[0]
        perm = torch.LongTensor(rng.permutation(mlp_dim))
        state[f"{pfx}.mlp.0.weight"] = state[f"{pfx}.mlp.0.weight"][perm]
        state[f"{pfx}.mlp.0.bias"]   = state[f"{pfx}.mlp.0.bias"][perm]
        state[f"{pfx}.mlp.3.weight"] = state[f"{pfx}.mlp.3.weight"][:, perm]

    permuted.load_state_dict(state)
    return permuted


# ──────────────────────────────────────────────────────────
# 5. SINGLE EXPERIMENT RUN
# ──────────────────────────────────────────────────────────

def run_experiment(heterogeneity='moderate', epochs=15, batch_size=128,
                   seed=42):
    """Train two clients, measure naive averaging degradation."""

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: heterogeneity = {heterogeneity}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── Data ──
    print("\n  Loading CIFAR-10 and creating splits...")
    train_a, train_b, testset = get_cifar10_splits(
        heterogeneity=heterogeneity)

    loader_a = DataLoader(train_a, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    loader_b = DataLoader(train_b, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Shared initialisation ──
    init_model = SmallViT()
    init_state = copy.deepcopy(init_model.state_dict())
    n_params = sum(p.numel() for p in init_model.parameters())
    print(f"  Model parameters: {n_params:,}")

    init_acc, _ = evaluate(init_model, test_loader, device)
    print(f"  Untrained baseline: {init_acc:.1f}%")

    # ── Train Client A ──
    print(f"\n  Training Client A ({epochs} epochs)...")
    model_a = SmallViT()
    model_a.load_state_dict(copy.deepcopy(init_state))
    losses_a = train_model(model_a, loader_a, device, epochs=epochs)

    # ── Train Client B ──
    print(f"\n  Training Client B ({epochs} epochs)...")
    model_b = SmallViT()
    model_b.load_state_dict(copy.deepcopy(init_state))
    losses_b = train_model(model_b, loader_b, device, epochs=epochs)

    # ── Evaluate everything ──
    print("\n  Evaluating all models on full test set...")
    acc_a, loss_a = evaluate(model_a, test_loader, device)
    acc_b, loss_b = evaluate(model_b, test_loader, device)
    print(f"  Client A accuracy:        {acc_a:.2f}%")
    print(f"  Client B accuracy:        {acc_b:.2f}%")

    # Naive average (the core question)
    avg_model = naive_average(model_a, model_b)
    acc_avg, loss_avg = evaluate(avg_model, test_loader, device)
    print(f"  Naive average accuracy:   {acc_avg:.2f}%")

    # Worst case: permute B then average (maximum misalignment)
    model_b_perm = permute_model(model_b, seed=123)
    acc_b_perm, _ = evaluate(model_b_perm, test_loader, device)
    print(f"  Client B permuted acc:    {acc_b_perm:.2f}%  "
          f"(sanity check — should ≈ Client B)")

    avg_perm = naive_average(model_a, model_b_perm)
    acc_perm_avg, loss_perm_avg = evaluate(avg_perm, test_loader, device)
    print(f"  Permuted average acc:     {acc_perm_avg:.2f}%  (worst case)")

    # ── Metrics ──
    best = max(acc_a, acc_b)
    results = {
        'heterogeneity':         heterogeneity,
        'epochs':                epochs,
        'untrained_acc':         round(init_acc, 2),
        'client_a_acc':          round(acc_a, 2),
        'client_b_acc':          round(acc_b, 2),
        'naive_avg_acc':         round(acc_avg, 2),
        'permuted_avg_acc':      round(acc_perm_avg, 2),
        'best_individual':       round(best, 2),
        'degradation_naive':     round(best - acc_avg, 2),
        'degradation_permuted':  round(best - acc_perm_avg, 2),
        'loss_a': round(loss_a, 4),
        'loss_b': round(loss_b, 4),
        'loss_avg': round(loss_avg, 4),
        'loss_perm_avg': round(loss_perm_avg, 4),
    }
    return results


# ──────────────────────────────────────────────────────────
# 6. REPORTING
# ──────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n{'='*74}")
    print(f"  SUMMARY — Permutation Invariance Problem in Federated Transformers")
    print(f"{'='*74}")
    print(f"  Model:   SmallViT  (4 layers, 4 heads, embed=128, ~1.2M params)")
    print(f"  Data:    CIFAR-10  (50K train, 10K test)")
    print(f"{'='*74}")
    print(f"  {'Het.':<10} | {'Cli A':>7} | {'Cli B':>7} | "
          f"{'Naive':>7} | {'Perm':>7} | {'Degrad.':>8} | {'Worst':>8}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*7}-+-"
          f"{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")
    for r in all_results:
        print(f"  {r['heterogeneity']:<10} | "
              f"{r['client_a_acc']:>6.1f}% | {r['client_b_acc']:>6.1f}% | "
              f"{r['naive_avg_acc']:>6.1f}% | {r['permuted_avg_acc']:>6.1f}% | "
              f"{r['degradation_naive']:>+7.1f}% | "
              f"{r['degradation_permuted']:>+7.1f}%")
    print()
    print("  Columns:")
    print("    Naive    = FedAvg (average raw weights directly)")
    print("    Perm     = average after randomly permuting one model's neurons")
    print("    Degrad.  = best individual − naive average  (+ means averaging hurts)")
    print("    Worst    = best individual − permuted average (upper bound on damage)")
    print()
    for r in all_results:
        d = r['degradation_naive']
        h = r['heterogeneity']
        if d > 10:
            verdict = "STRONG signal — clear motivation for alignment"
        elif d > 3:
            verdict = "MODERATE signal — alignment likely helps"
        elif d > 0:
            verdict = "MILD signal — alignment may help at scale"
        else:
            verdict = "NO signal — naive averaging works fine here"
        print(f"  {h}: {verdict}  ({d:+.1f}%)")


def save_plot(all_results, path='experiment1_results.pdf'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        labels = ['Client A', 'Client B', 'Naive\nAverage', 'Permuted\nAverage']
        accs   = [r['client_a_acc'], r['client_b_acc'],
                  r['naive_avg_acc'], r['permuted_avg_acc']]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

        bars = ax.bar(labels, accs, color=colors, edgecolor='white',
                      linewidth=1.5, width=0.6)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=11)

        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Heterogeneity: {r["heterogeneity"]}',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(accs) * 1.18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        d = r['degradation_naive']
        ax.annotate(f'Degradation: {d:+.1f}%', xy=(0.5, 0.02),
                    xycoords='axes fraction', ha='center', fontsize=11,
                    color='red' if d > 3 else 'orange', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              edgecolor='gray', alpha=0.8))

    fig.suptitle('Experiment 1: Permutation Invariance Problem\n'
                 'in Federated Transformer Weight Averaging',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved → {path}")


# ──────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()

    all_results = []
    for het in ['mild', 'moderate', 'extreme']:
        r = run_experiment(heterogeneity=het, epochs=15,
                           batch_size=128, seed=42)
        all_results.append(r)

    print_summary(all_results)
    save_plot(all_results, 'experiment1_results.pdf')

    with open('experiment1_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON saved → experiment1_results.json")
    print(f"\n  Total wall time: {(time.time() - t0) / 60:.1f} minutes")
