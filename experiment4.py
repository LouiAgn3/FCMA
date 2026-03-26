"""
Experiment 4: Knowledge Distillation-Based Federated Averaging (FedDF)
======================================================================
Goal: Since weight-space alignment fails (Experiments 2-3), test whether
      averaging in PREDICTION SPACE works better.

Instead of averaging weights (FedAvg), FedDF does:
  1. Each client sends their trained model to the server
  2. Server runs a public reference dataset through ALL client models
  3. Server averages the soft predictions (logits) — this is the ensemble
  4. Server trains a fresh/warm-started model to match the ensemble
     predictions via KL-divergence (knowledge distillation)

Why this should work:
  - Weight averaging fails because models have different internal
    representations that can't be aligned geometrically
  - But their OUTPUTS (predictions) are in a shared, meaningful space
    — class probabilities — that doesn't have permutation ambiguity
  - Distillation extracts the collective knowledge without requiring
    internal alignment

We compare:
  1. Individual models (upper bound reference)
  2. Ensemble (test-time logit averaging — theoretical ceiling)
  3. Naive average (FedAvg — the broken baseline)
  4. Weight-matched average (best alignment method from Exp 2-3)
  5. FedDF cold — distillation from fresh init
  6. FedDF warm — distillation starting from naive average

Sweep over [1, 2, 3, 5, 10, 15] local epochs as in Experiment 3.

Run:
  pip install torch torchvision matplotlib scipy
  python experiment4.py

  GPU: ~40-60 min  |  CPU: ~4-6 hours

Output:
  - experiment4_crossover.pdf   (accuracy vs local epochs)
  - experiment4_recovery.pdf    (recovery % vs local epochs)
  - experiment4_results.json    (raw numbers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy
import json
import time
from scipy.optimize import linear_sum_assignment


# ══════════════════════════════════════════════════════════
# 1. MODEL (same SmallViT)
# ══════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=128, depth=4, num_heads=4,
                 mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
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
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))


# ══════════════════════════════════════════════════════════
# 2. DATA
# ══════════════════════════════════════════════════════════

def get_cifar10_splits(data_dir='./data', heterogeneity='moderate'):
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

    if heterogeneity == 'mild':
        idx = np.random.permutation(len(targets))
        s = len(targets) // 2
        a_idx, b_idx = idx[:s], idx[s:]
    elif heterogeneity == 'moderate':
        a_idx, b_idx = [], []
        for c in range(10):
            ci = np.where(targets == c)[0]
            np.random.shuffle(ci)
            p = np.random.dirichlet([0.5, 0.5])
            s = int(len(ci) * p[0])
            a_idx.extend(ci[:s]); b_idx.extend(ci[s:])
        a_idx, b_idx = np.array(a_idx), np.array(b_idx)
    elif heterogeneity == 'extreme':
        a_idx = np.where(targets < 5)[0]
        b_idx = np.where(targets >= 5)[0]

    return Subset(trainset, a_idx), Subset(trainset, b_idx), testset


def get_reference_dataset(testset, n_samples=2000):
    """Public reference dataset for distillation (larger than matching)."""
    indices = np.random.choice(len(testset), n_samples, replace=False)
    return Subset(testset, indices)


# ══════════════════════════════════════════════════════════
# 3. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════

def train_model(model, train_loader, device, epochs=15, lr=1e-3):
    model.to(device); model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            _, pred = model(inputs).max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        scheduler.step()
        if epochs <= 5 or (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Acc: {100.*correct/total:.1f}%")


def evaluate(model, test_loader, device):
    model.to(device); model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, pred = model(inputs).max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100. * correct / total


def evaluate_ensemble(models, test_loader, device):
    """Test-time ensemble — theoretical upper bound for distillation."""
    for m in models: m.to(device); m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            avg_logits = sum(m(inputs) for m in models) / len(models)
            _, pred = avg_logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100. * correct / total


# ══════════════════════════════════════════════════════════
# 4. FEDAVG & WEIGHT MATCHING (baselines)
# ══════════════════════════════════════════════════════════

def naive_average(model_a, model_b):
    avg = copy.deepcopy(model_a)
    sa, sb = model_a.state_dict(), model_b.state_dict()
    avg.load_state_dict({k: (sa[k] + sb[k]) / 2.0 for k in sa})
    return avg


def apply_head_permutation(state, blk, perm, E, H):
    hd = E // H; pfx = f"blocks.{blk}"
    iw = state[f"{pfx}.attn.in_proj_weight"].clone()
    ib = state[f"{pfx}.attn.in_proj_bias"].clone()
    ow = state[f"{pfx}.attn.out_proj.weight"].clone()
    niw, nib, now = iw.clone(), ib.clone(), ow.clone()
    for qkv in range(3):
        o = qkv * E
        for nh, oh in enumerate(perm):
            s = slice(o+oh*hd, o+oh*hd+hd); d = slice(o+nh*hd, o+nh*hd+hd)
            niw[d] = iw[s]; nib[d] = ib[s]
    for nh, oh in enumerate(perm):
        s = slice(oh*hd, oh*hd+hd); d = slice(nh*hd, nh*hd+hd)
        now[:, d] = ow[:, s]
    state[f"{pfx}.attn.in_proj_weight"] = niw
    state[f"{pfx}.attn.in_proj_bias"] = nib
    state[f"{pfx}.attn.out_proj.weight"] = now


def apply_ffn_permutation(state, blk, perm):
    pfx = f"blocks.{blk}"; pi = torch.LongTensor(perm)
    state[f"{pfx}.mlp.0.weight"] = state[f"{pfx}.mlp.0.weight"][pi]
    state[f"{pfx}.mlp.0.bias"]   = state[f"{pfx}.mlp.0.bias"][pi]
    state[f"{pfx}.mlp.3.weight"] = state[f"{pfx}.mlp.3.weight"][:, pi]


def weight_matched_average(model_ref, model_b):
    sr = model_ref.state_dict()
    sb = copy.deepcopy(model_b.state_dict())
    nblk = len(model_ref.blocks)
    E = model_ref.blocks[0].attn.embed_dim
    H = model_ref.blocks[0].attn.num_heads
    hd = E // H
    for blk in range(nblk):
        pfx = f"blocks.{blk}"
        cost = np.zeros((H, H))
        for i in range(H):
            fr = []
            for qkv in range(3):
                o = qkv*E; s = slice(o+i*hd, o+i*hd+hd)
                fr.append(sr[f"{pfx}.attn.in_proj_weight"][s].flatten())
                fr.append(sr[f"{pfx}.attn.in_proj_bias"][s])
            fr.append(sr[f"{pfx}.attn.out_proj.weight"][:, i*hd:i*hd+hd].flatten())
            fr = torch.cat(fr).cpu().numpy()
            for j in range(H):
                fb = []
                for qkv in range(3):
                    o = qkv*E; s = slice(o+j*hd, o+j*hd+hd)
                    fb.append(sb[f"{pfx}.attn.in_proj_weight"][s].flatten())
                    fb.append(sb[f"{pfx}.attn.in_proj_bias"][s])
                fb.append(sb[f"{pfx}.attn.out_proj.weight"][:, j*hd:j*hd+hd].flatten())
                fb = torch.cat(fb).cpu().numpy()
                cost[i, j] = np.linalg.norm(fr - fb)
        _, ci = linear_sum_assignment(cost)
        apply_head_permutation(sb, blk, ci, E, H)
        ur = sr[f"{pfx}.mlp.0.weight"]; ubr = sr[f"{pfx}.mlp.0.bias"]
        dr = sr[f"{pfx}.mlp.3.weight"]
        ub_ = sb[f"{pfx}.mlp.0.weight"]; ubb = sb[f"{pfx}.mlp.0.bias"]
        db = sb[f"{pfx}.mlp.3.weight"]
        fpr = torch.cat([ur, ubr.unsqueeze(1), dr.t()], 1).cpu().numpy()
        fpb = torch.cat([ub_, ubb.unsqueeze(1), db.t()], 1).cpu().numpy()
        rsq = np.sum(fpr**2, 1, keepdims=True)
        bsq = np.sum(fpb**2, 1, keepdims=True)
        dsq = np.maximum(rsq + bsq.T - 2*fpr@fpb.T, 0)
        _, ci = linear_sum_assignment(np.sqrt(dsq))
        apply_ffn_permutation(sb, blk, ci)
    ab = copy.deepcopy(model_ref); ab.load_state_dict(sb)
    return naive_average(model_ref, ab)


# ══════════════════════════════════════════════════════════
# 5. FedDF — KNOWLEDGE DISTILLATION
# ══════════════════════════════════════════════════════════

def generate_ensemble_logits(models, ref_dataset, device, batch_size=64):
    """Average logits from all models on reference data."""
    loader = DataLoader(ref_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    all_imgs, all_logits = [], [[] for _ in models]
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            all_imgs.append(inputs.cpu())
            for i, m in enumerate(models):
                m.to(device); m.eval()
                all_logits[i].append(m(inputs).cpu())
    imgs = torch.cat(all_imgs, dim=0)
    avg = torch.stack([torch.cat(al, 0) for al in all_logits]).mean(0)
    return imgs, avg


def distill_from_ensemble(student, images, teacher_logits, device,
                          epochs=20, lr=1e-3, temperature=3.0,
                          batch_size=64):
    """Train student to match ensemble soft predictions via KL divergence."""
    student.to(device); student.train()
    loader = DataLoader(TensorDataset(images, teacher_logits),
                        batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        for img, tgt in loader:
            img, tgt = img.to(device), tgt.to(device)
            optimizer.zero_grad()
            s_logits = student(img)
            # KL divergence on softened distributions
            t_soft = F.softmax(tgt / temperature, dim=1)
            s_log_soft = F.log_softmax(s_logits / temperature, dim=1)
            loss = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temperature ** 2)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epochs <= 10 or (epoch+1) % 5 == 0 or epoch == 0:
            print(f"      Distill {epoch+1:2d}/{epochs} | "
                  f"KL: {total_loss/len(loader):.4f}")
    return student


def feddf_average(models, ref_dataset, init_state, device,
                  distill_epochs=20, temperature=3.0, warm_start=None):
    """
    FedDF: distill ensemble knowledge into a single model.
    warm_start: optional state_dict to initialize student (e.g. naive avg).
    """
    print("    Generating ensemble logits...")
    images, avg_logits = generate_ensemble_logits(models, ref_dataset, device)

    student = SmallViT()
    if warm_start is not None:
        student.load_state_dict(copy.deepcopy(warm_start))
        start_type = "warm (from naive avg)"
    else:
        student.load_state_dict(copy.deepcopy(init_state))
        start_type = "cold (from shared init)"

    print(f"    Distilling [{start_type}] "
          f"({distill_epochs} ep, T={temperature})...")
    student = distill_from_ensemble(
        student, images, avg_logits, device,
        epochs=distill_epochs, temperature=temperature)
    return student


# ══════════════════════════════════════════════════════════
# 6. EPOCH SWEEP
# ══════════════════════════════════════════════════════════

def run_epoch_sweep(heterogeneity='moderate', epoch_list=None,
                    batch_size=128, seed=42):
    if epoch_list is None:
        epoch_list = [1, 2, 3, 5, 10, 15]

    print(f"\n{'#'*65}")
    print(f"  EPOCH SWEEP: heterogeneity = {heterogeneity}")
    print(f"  Epochs: {epoch_list}")
    print(f"{'#'*65}")

    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    train_a, train_b, testset = get_cifar10_splits(heterogeneity=heterogeneity)
    loader_a = DataLoader(train_a, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    loader_b = DataLoader(train_b, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    ref_dataset = get_reference_dataset(testset, n_samples=2000)

    init_model = SmallViT()
    init_state = copy.deepcopy(init_model.state_dict())

    sweep_results = []

    for epochs in epoch_list:
        print(f"\n  {'─'*55}")
        print(f"  Local epochs: {epochs}")
        print(f"  {'─'*55}")

        torch.manual_seed(seed); np.random.seed(seed)

        # Train clients
        model_a = SmallViT()
        model_a.load_state_dict(copy.deepcopy(init_state))
        train_model(model_a, loader_a, device, epochs=epochs)

        model_b = SmallViT()
        model_b.load_state_dict(copy.deepcopy(init_state))
        train_model(model_b, loader_b, device, epochs=epochs)

        acc_a = evaluate(model_a, test_loader, device)
        acc_b = evaluate(model_b, test_loader, device)
        best = max(acc_a, acc_b)
        print(f"  Client A: {acc_a:.2f}% | Client B: {acc_b:.2f}%")

        # Ensemble
        acc_ens = evaluate_ensemble([model_a, model_b], test_loader, device)
        print(f"  Ensemble: {acc_ens:.2f}%")

        # Naive avg
        naive_model = naive_average(model_a, model_b)
        acc_naive = evaluate(naive_model, test_loader, device)
        print(f"  Naive FedAvg: {acc_naive:.2f}%")

        # Weight-matched
        acc_wm = evaluate(weight_matched_average(model_a, model_b),
                          test_loader, device)
        print(f"  Weight-matched: {acc_wm:.2f}%")

        # FedDF cold
        print("\n  [FedDF cold start]")
        m_cold = feddf_average([model_a, model_b], ref_dataset,
                               init_state, device,
                               distill_epochs=20, temperature=3.0)
        acc_cold = evaluate(m_cold, test_loader, device)
        print(f"  FedDF cold: {acc_cold:.2f}%")

        # FedDF warm
        print("\n  [FedDF warm start]")
        m_warm = feddf_average([model_a, model_b], ref_dataset,
                               init_state, device,
                               distill_epochs=20, temperature=3.0,
                               warm_start=naive_model.state_dict())
        acc_warm = evaluate(m_warm, test_loader, device)
        print(f"  FedDF warm: {acc_warm:.2f}%")

        # FedDF warm T=5
        print("\n  [FedDF warm, T=5]")
        m_t5 = feddf_average([model_a, model_b], ref_dataset,
                             init_state, device,
                             distill_epochs=20, temperature=5.0,
                             warm_start=naive_model.state_dict())
        acc_t5 = evaluate(m_t5, test_loader, device)
        print(f"  FedDF warm T=5: {acc_t5:.2f}%")

        gap = best - acc_naive
        best_feddf = max(acc_cold, acc_warm, acc_t5)

        sweep_results.append({
            'epochs': epochs,
            'client_a': round(acc_a, 2),
            'client_b': round(acc_b, 2),
            'best_individual': round(best, 2),
            'ensemble': round(acc_ens, 2),
            'naive': round(acc_naive, 2),
            'weight_matched': round(acc_wm, 2),
            'feddf_cold': round(acc_cold, 2),
            'feddf_warm': round(acc_warm, 2),
            'feddf_t5': round(acc_t5, 2),
            'best_feddf': round(best_feddf, 2),
            'gap': round(gap, 2),
            'recovery_wm': round((acc_wm - acc_naive) / (gap + 1e-10) * 100, 1),
            'recovery_cold': round((acc_cold - acc_naive) / (gap + 1e-10) * 100, 1),
            'recovery_warm': round((acc_warm - acc_naive) / (gap + 1e-10) * 100, 1),
            'recovery_best': round((best_feddf - acc_naive) / (gap + 1e-10) * 100, 1),
            'distill_gap': round(acc_ens - best_feddf, 2),
        })

    return {'heterogeneity': heterogeneity, 'sweeps': sweep_results}


# ══════════════════════════════════════════════════════════
# 7. REPORTING
# ══════════════════════════════════════════════════════════

def print_summary(all_sweeps):
    for sweep in all_sweeps:
        het = sweep['heterogeneity']
        results = sweep['sweeps']
        print(f"\n{'='*92}")
        print(f"  {het.upper()} — FedDF vs FedAvg vs Weight Matching")
        print(f"{'='*92}")
        print(f"  {'Ep':>3} | {'Best':>6} | {'Ensem':>6} | {'Naive':>6} | "
              f"{'W-Mat':>6} | {'FedDF':>6} | {'FedDF':>6} | {'FedDF':>6} | "
              f"{'Gap':>6} | {'Best':>5}")
        print(f"  {'':>3} | {'Indv':>6} | {'':>6} | {'Avg':>6} | "
              f"{'':>6} | {'cold':>6} | {'warm':>6} | {'T=5':>6} | "
              f"{'':>6} | {'D-%':>5}")
        print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*6}-+-{'-'*5}")
        for r in results:
            print(f"  {r['epochs']:>3} | "
                  f"{r['best_individual']:>5.1f}% | "
                  f"{r['ensemble']:>5.1f}% | "
                  f"{r['naive']:>5.1f}% | "
                  f"{r['weight_matched']:>5.1f}% | "
                  f"{r['feddf_cold']:>5.1f}% | "
                  f"{r['feddf_warm']:>5.1f}% | "
                  f"{r['feddf_t5']:>5.1f}% | "
                  f"{r['gap']:>+5.1f}% | "
                  f"{r['recovery_best']:>4.0f}%")
    print(f"\n  Key:")
    print(f"    Ensem    = test-time ensemble (theoretical ceiling)")
    print(f"    FedDF cold/warm = distillation from init / from naive avg")
    print(f"    Best D-% = best FedDF variant recovery percentage")


def save_plot(all_sweeps, path='experiment4_crossover.pdf'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(all_sweeps)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5.5))
    if n == 1: axes = [axes]

    for ax, sweep in zip(axes, all_sweeps):
        het = sweep['heterogeneity']
        R = sweep['sweeps']
        ep = [r['epochs'] for r in R]

        ax.plot(ep, [r['ensemble'] for r in R], '*-', color='#00BCD4',
                linewidth=2, markersize=10, label='Ensemble')
        ax.plot(ep, [r['best_individual'] for r in R], 'o-', color='#2196F3',
                linewidth=2, markersize=7, label='Best Individual')
        ax.plot(ep, [r['naive'] for r in R], 's--', color='#FF9800',
                linewidth=2, markersize=7, label='Naive FedAvg')
        ax.plot(ep, [r['weight_matched'] for r in R], '^--', color='#9C27B0',
                linewidth=1.5, markersize=6, label='Weight-Matched', alpha=0.7)
        ax.plot(ep, [r['feddf_cold'] for r in R], 'D-', color='#F44336',
                linewidth=2, markersize=7, label='FedDF cold')
        ax.plot(ep, [r['feddf_warm'] for r in R], 'p-', color='#4CAF50',
                linewidth=2.5, markersize=9, label='FedDF warm')

        ax.fill_between(ep, [r['naive'] for r in R],
                        [r['best_individual'] for r in R],
                        alpha=0.08, color='#FF9800')
        ax.set_xlabel('Local Training Epochs', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Heterogeneity: {het}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_xticks(ep)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 4: Knowledge Distillation (FedDF)\n'
                 'vs Weight-Space Methods',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → {path}")


def save_recovery_plot(all_sweeps, path='experiment4_recovery.pdf'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(all_sweeps)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1: axes = [axes]

    for ax, sweep in zip(axes, all_sweeps):
        het = sweep['heterogeneity']
        R = sweep['sweeps']
        ep = [r['epochs'] for r in R]

        ax.plot(ep, [r['recovery_wm'] for r in R], '^--', color='#9C27B0',
                linewidth=1.5, markersize=6, label='Weight-Matched', alpha=0.7)
        ax.plot(ep, [r['recovery_cold'] for r in R], 'D-', color='#F44336',
                linewidth=2, markersize=7, label='FedDF cold')
        ax.plot(ep, [r['recovery_warm'] for r in R], 'p-', color='#4CAF50',
                linewidth=2.5, markersize=9, label='FedDF warm')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel('Local Training Epochs', fontsize=12)
        ax.set_ylabel('Recovery (%)', fontsize=12)
        ax.set_title(f'Heterogeneity: {het}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks(ep)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 4: Recovery (%) — FedDF vs Weight Matching',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Recovery plot saved → {path}")


# ══════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()
    all_sweeps = []
    for het in ['mild', 'moderate', 'extreme']:
        sweep = run_epoch_sweep(heterogeneity=het,
                                epoch_list=[1, 2, 3, 5, 10, 15],
                                batch_size=128, seed=42)
        all_sweeps.append(sweep)

    print_summary(all_sweeps)
    save_plot(all_sweeps, 'experiment4_crossover.pdf')
    save_recovery_plot(all_sweeps, 'experiment4_recovery.pdf')

    with open('experiment4_results.json', 'w') as f:
        json.dump(all_sweeps, f, indent=2)
    print(f"  JSON saved → experiment4_results.json")
    print(f"\n  Total wall time: {(time.time() - t0) / 60:.1f} minutes")
