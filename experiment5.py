"""
Experiment 5: Layer-Adaptive Hybrid Aggregation (FedAD)
========================================================
Goal: Show that adaptively choosing between weight averaging and
      distillation on a per-layer basis, guided by Procrustes quality,
      outperforms both pure FedAvg and pure FedDF.

Two novel components:
  1. Layer-adaptive hybrid: weight-average layers with high Procrustes
     quality, distill layers with low quality
  2. Confidence-weighted ensemble: weight each client's logits per-sample
     by inverse entropy (confident predictions get more weight)

We compare:
  - Naive average (FedAvg baseline)
  - Weight-matched average (Exp 2 baseline)
  - FedDF warm-start (Exp 4 baseline)
  - FedAD: layer-adaptive hybrid
  - FedAD+CW: hybrid + confidence weighting

Run:
  pip install torch torchvision matplotlib scipy
  python experiment5.py

  GPU: ~40-50 min  |  CPU: ~4-5 hours
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
# 1. MODEL
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
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout))
    def forward(self, x):
        h = self.norm1(x); h, _ = self.attn(h, h, h); x = x + h
        x = x + self.mlp(self.norm2(x)); return x


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
            for _ in range(depth)])
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
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]; x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks: x = block(x)
        return self.head(self.norm(x[:, 0]))

    def forward_features(self, x):
        """Forward pass returning intermediate representations at each block."""
        B = x.shape[0]; x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        intermediates = []
        for block in self.blocks:
            x = block(x)
            intermediates.append(x)
        return self.head(self.norm(x[:, 0])), intermediates


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
        indices = np.random.permutation(len(targets))
        split = len(targets) // 2
        a_idx, b_idx = indices[:split], indices[split:]
    elif heterogeneity == 'moderate':
        a_idx, b_idx = [], []
        for c in range(10):
            ci = np.where(targets == c)[0]; np.random.shuffle(ci)
            props = np.random.dirichlet([0.5, 0.5])
            s = int(len(ci) * props[0])
            a_idx.extend(ci[:s]); b_idx.extend(ci[s:])
        a_idx, b_idx = np.array(a_idx), np.array(b_idx)
    elif heterogeneity == 'extreme':
        a_idx = np.where(targets < 5)[0]
        b_idx = np.where(targets >= 5)[0]

    print(f"  Split: {heterogeneity} | A: {len(a_idx)} | B: {len(b_idx)}")
    return Subset(trainset, a_idx), Subset(trainset, b_idx), testset


def get_reference_dataset(testset, n_samples=500):
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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0); correct += pred.eq(labels).sum().item()
        scheduler.step()
        if epochs <= 5 or (epoch + 1) % 5 == 0 or epoch == 0:
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
            total += labels.size(0); correct += pred.eq(labels).sum().item()
    return 100. * correct / total


# ══════════════════════════════════════════════════════════
# 4. BASELINES
# ══════════════════════════════════════════════════════════

def naive_average(model_a, model_b):
    averaged = copy.deepcopy(model_a)
    sa, sb = model_a.state_dict(), model_b.state_dict()
    averaged.load_state_dict({k: (sa[k] + sb[k]) / 2.0 for k in sa})
    return averaged


# ══════════════════════════════════════════════════════════
# 5. PROCRUSTES QUALITY MEASUREMENT
# ══════════════════════════════════════════════════════════

def collect_layer_outputs(model, ref_loader, device):
    """Collect residual stream output after each block."""
    model.to(device); model.eval()
    layer_outs = {i: [] for i in range(len(model.blocks))}
    hooks = []
    for i, block in enumerate(model.blocks):
        def make_hook(idx):
            def hook_fn(module, input, output):
                layer_outs[idx].append(output.detach())
            return hook_fn
        hooks.append(block.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for inputs, _ in ref_loader:
            model(inputs.to(device))
    for h in hooks: h.remove()
    result = {}
    for idx, outs in layer_outs.items():
        all_out = torch.cat(outs, dim=0)
        result[idx] = all_out.reshape(-1, all_out.shape[-1]).cpu()
    return result


def compute_procrustes_quality(X_ref, X_b):
    """Compute Procrustes quality score between two activation matrices."""
    X_ref_c = X_ref - X_ref.mean(0)
    X_b_c = X_b - X_b.mean(0)
    M = X_ref_c.T @ X_b_c
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0] * (U.shape[1] - 1) + [np.sign(d)])
    R = Vt.T @ D @ U.T
    aligned = X_b_c @ R
    residual = np.linalg.norm(X_ref_c - aligned, 'fro')
    total = np.linalg.norm(X_ref_c, 'fro')
    quality = 1.0 - (residual / (total + 1e-10))
    return quality


def measure_layer_qualities(model_a, model_b, ref_loader, device):
    """Compute Procrustes quality at each layer."""
    outs_a = collect_layer_outputs(model_a, ref_loader, device)
    outs_b = collect_layer_outputs(model_b, ref_loader, device)
    qualities = {}
    for blk in range(len(model_a.blocks)):
        q = compute_procrustes_quality(
            outs_a[blk].numpy(), outs_b[blk].numpy())
        qualities[blk] = q
    return qualities


# ══════════════════════════════════════════════════════════
# 6. STANDARD FedDF (warm-start baseline from Exp 4)
# ══════════════════════════════════════════════════════════

def generate_soft_labels(models, ref_loader, device, temperature=3.0):
    for m in models: m.to(device); m.eval()
    all_inputs, all_logits = [], []
    with torch.no_grad():
        for inputs, _ in ref_loader:
            inputs = inputs.to(device)
            logits_list = [m(inputs) for m in models]
            avg_logits = torch.stack(logits_list).mean(dim=0)
            all_inputs.append(inputs.cpu())
            all_logits.append(avg_logits.cpu())
    return torch.cat(all_inputs), torch.cat(all_logits)


def generate_confidence_weighted_labels(models, ref_loader, device,
                                         temperature=3.0):
    """
    Weight each client's logits per-sample by inverse entropy.
    High confidence (low entropy) → high weight.
    """
    for m in models: m.to(device); m.eval()
    all_inputs, all_logits = [], []
    with torch.no_grad():
        for inputs, _ in ref_loader:
            inputs = inputs.to(device)
            logits_list = [m(inputs) for m in models]

            # Compute per-sample entropy for each model
            weights = []
            for logits in logits_list:
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                # Inverse entropy as confidence weight
                # Lower entropy = more confident = higher weight
                max_entropy = np.log(probs.shape[1])  # log(num_classes)
                confidence = 1.0 - (entropy / max_entropy)  # [0, 1]
                weights.append(confidence)

            # Stack: (num_models, batch_size)
            weights = torch.stack(weights, dim=0)
            # Normalise so weights sum to 1 per sample
            weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-10)

            # Weighted average of logits
            # logits_list: list of (batch, classes)
            # weights: (num_models, batch)
            stacked = torch.stack(logits_list, dim=0)  # (M, B, C)
            w = weights.unsqueeze(-1)  # (M, B, 1)
            weighted_logits = (stacked * w).sum(dim=0)  # (B, C)

            all_inputs.append(inputs.cpu())
            all_logits.append(weighted_logits.cpu())

    return torch.cat(all_inputs), torch.cat(all_logits)


def distill_model(student, ref_inputs, ref_logits, device,
                  init_state=None, epochs=20, lr=1e-3, temperature=3.0,
                  batch_size=64):
    if init_state is not None:
        student.load_state_dict(copy.deepcopy(init_state))
    student.to(device); student.train()
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = TensorDataset(ref_inputs, ref_logits)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0
        for inputs, teacher_logits in loader:
            inputs, teacher_logits = inputs.to(device), teacher_logits.to(device)
            optimizer.zero_grad()
            student_logits = student(inputs)
            teacher_soft = F.log_softmax(teacher_logits / temperature, dim=1)
            student_soft = F.log_softmax(student_logits / temperature, dim=1)
            loss = F.kl_div(student_soft, teacher_soft.exp(),
                           reduction='batchmean') * (temperature ** 2)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epochs <= 10 or (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Distill {epoch+1:2d}/{epochs} | "
                  f"KL: {total_loss/len(loader):.4f}")
    return student


def feddf_warm(models, ref_dataset, naive_state, device,
               distill_epochs=20, temperature=3.0):
    ref_loader = DataLoader(ref_dataset, batch_size=64,
                            shuffle=False, num_workers=2, pin_memory=True)
    print("    Generating uniform ensemble logits...")
    ref_inputs, avg_logits = generate_soft_labels(
        models, ref_loader, device, temperature)
    print("    Distilling (warm start from naive avg)...")
    student = SmallViT()
    student = distill_model(student, ref_inputs, avg_logits, device,
                            init_state=naive_state, epochs=distill_epochs,
                            lr=5e-4, temperature=temperature)
    return student


# ══════════════════════════════════════════════════════════
# 7. FedAD: LAYER-ADAPTIVE HYBRID AGGREGATION (NEW METHOD)
# ══════════════════════════════════════════════════════════

def build_hybrid_state(model_a, model_b, qualities, threshold=0.4):
    """
    Build a hybrid model state dict:
      - For blocks with quality >= threshold: weight-average
      - For blocks with quality < threshold: keep model_a's weights
        (these will be trained via distillation)

    Also average embedding layers and classification head (shared structure).
    """
    sa = model_a.state_dict()
    sb = model_b.state_dict()
    hybrid = {}

    avg_blocks = []
    distill_blocks = []

    for key in sa:
        # Determine which block this key belongs to
        block_idx = None
        for i in range(len(model_a.blocks)):
            if key.startswith(f"blocks.{i}."):
                block_idx = i
                break

        if block_idx is not None:
            if qualities[block_idx] >= threshold:
                # High quality: weight-average this block
                hybrid[key] = (sa[key] + sb[key]) / 2.0
                if block_idx not in avg_blocks:
                    avg_blocks.append(block_idx)
            else:
                # Low quality: keep naive average (will be refined by distillation)
                hybrid[key] = (sa[key] + sb[key]) / 2.0
                if block_idx not in distill_blocks:
                    distill_blocks.append(block_idx)
        else:
            # Embedding, position, cls_token, norm, head: always average
            hybrid[key] = (sa[key] + sb[key]) / 2.0

    print(f"    Averaged blocks (quality >= {threshold}): {sorted(avg_blocks)}")
    print(f"    Distilled blocks (quality < {threshold}): {sorted(distill_blocks)}")

    return hybrid, sorted(distill_blocks)


def collect_teacher_intermediates(models, ref_loader, device, block_indices):
    """
    Collect intermediate activations from teacher ensemble at specific blocks.
    Returns averaged activations across models.
    """
    for m in models: m.to(device); m.eval()

    all_inputs = []
    block_acts = {idx: [] for idx in block_indices}
    all_logits = []

    with torch.no_grad():
        for inputs, _ in ref_loader:
            inputs = inputs.to(device)
            all_inputs.append(inputs.cpu())

            batch_logits = []
            batch_acts = {idx: [] for idx in block_indices}

            for m in models:
                logits, intermediates = m.forward_features(inputs)
                batch_logits.append(logits)
                for idx in block_indices:
                    batch_acts[idx].append(intermediates[idx])

            # Average logits across models
            all_logits.append(torch.stack(batch_logits).mean(0).cpu())

            # Average intermediates across models
            for idx in block_indices:
                avg_act = torch.stack(batch_acts[idx]).mean(0)
                block_acts[idx].append(avg_act.cpu())

    all_inputs = torch.cat(all_inputs)
    all_logits = torch.cat(all_logits)
    for idx in block_indices:
        block_acts[idx] = torch.cat(block_acts[idx])

    return all_inputs, all_logits, block_acts


def distill_hybrid(student, ref_inputs, ref_logits, teacher_block_acts,
                   distill_block_indices, device, init_state=None,
                   epochs=20, lr=5e-4, temperature=3.0, batch_size=64,
                   hint_weight=1.0):
    """
    Hybrid distillation:
      - KL divergence loss on final logits (standard distillation)
      - MSE loss on intermediate representations at distilled blocks
        (hint-based distillation)

    The hint loss ensures the distilled blocks learn representations
    compatible with the averaged blocks above and below them.
    """
    if init_state is not None:
        student.load_state_dict(copy.deepcopy(init_state))
    student.to(device); student.train()
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build dataset including intermediate targets
    tensors = [ref_inputs, ref_logits]
    for idx in sorted(distill_block_indices):
        tensors.append(teacher_block_acts[idx])
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_extra = len(distill_block_indices)

    for epoch in range(epochs):
        total_kl, total_hint, total_loss = 0, 0, 0
        for batch in loader:
            inputs = batch[0].to(device)
            teacher_logits = batch[1].to(device)
            teacher_hints = {}
            for i, idx in enumerate(sorted(distill_block_indices)):
                teacher_hints[idx] = batch[2 + i].to(device)

            optimizer.zero_grad()
            student_logits, student_intermediates = student.forward_features(inputs)

            # KL divergence on final logits
            t_soft = F.log_softmax(teacher_logits / temperature, dim=1)
            s_soft = F.log_softmax(student_logits / temperature, dim=1)
            kl_loss = F.kl_div(s_soft, t_soft.exp(),
                              reduction='batchmean') * (temperature ** 2)

            # MSE on intermediate representations (hint loss)
            hint_loss = 0
            for idx in distill_block_indices:
                student_act = student_intermediates[idx]
                teacher_act = teacher_hints[idx]
                hint_loss = hint_loss + F.mse_loss(student_act, teacher_act)

            if n_extra > 0:
                hint_loss = hint_loss / n_extra

            loss = kl_loss + hint_weight * hint_loss
            loss.backward(); optimizer.step()

            total_kl += kl_loss.item()
            total_hint += hint_loss.item() if isinstance(hint_loss, torch.Tensor) else hint_loss
            total_loss += loss.item()

        scheduler.step()
        if epochs <= 10 or (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Distill {epoch+1:2d}/{epochs} | "
                  f"KL: {total_kl/len(loader):.4f} | "
                  f"Hint: {total_hint/len(loader):.4f} | "
                  f"Total: {total_loss/len(loader):.4f}")

    return student


def fedad_aggregate(models, ref_dataset, qualities, device,
                    threshold=0.4, distill_epochs=20, temperature=3.0,
                    hint_weight=1.0, use_confidence_weighting=False):
    """
    FedAD: Layer-Adaptive Hybrid Aggregation

    1. Measure Procrustes quality per layer
    2. Weight-average blocks with quality >= threshold
    3. For blocks below threshold, distill from ensemble
    4. Optionally use confidence-weighted ensemble
    """
    ref_loader = DataLoader(ref_dataset, batch_size=64,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Build hybrid init state
    hybrid_state, distill_blocks = build_hybrid_state(
        models[0], models[1], qualities, threshold)

    if len(distill_blocks) == 0:
        print("    All blocks above threshold — pure weight averaging")
        student = SmallViT()
        student.load_state_dict(hybrid_state)
        return student

    # Collect teacher targets
    print(f"    Collecting teacher intermediates for blocks {distill_blocks}...")
    if use_confidence_weighting:
        print("    Using confidence-weighted ensemble...")
        ref_inputs, avg_logits = generate_confidence_weighted_labels(
            models, ref_loader, device, temperature)
        # For intermediates, still use uniform average (confidence is logit-level)
        _, _, teacher_acts = collect_teacher_intermediates(
            models, ref_loader, device, distill_blocks)
    else:
        ref_inputs, avg_logits, teacher_acts = collect_teacher_intermediates(
            models, ref_loader, device, distill_blocks)

    # Distill with hybrid loss
    print(f"    Hybrid distillation ({distill_epochs} epochs, "
          f"hint_weight={hint_weight})...")
    student = SmallViT()
    student = distill_hybrid(
        student, ref_inputs, avg_logits, teacher_acts,
        distill_blocks, device, init_state=hybrid_state,
        epochs=distill_epochs, lr=5e-4, temperature=temperature,
        hint_weight=hint_weight)

    return student


# ══════════════════════════════════════════════════════════
# 8. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════

def run_experiment(heterogeneity='moderate', epochs=15, batch_size=128,
                   seed=42, distill_epochs=20, ref_size=500,
                   temperature=3.0, threshold=0.4):

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT 5: het={heterogeneity}, epochs={epochs}, "
          f"tau={threshold}")
    print(f"{'='*65}")

    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_a, train_b, testset = get_cifar10_splits(heterogeneity=heterogeneity)
    loader_a = DataLoader(train_a, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    loader_b = DataLoader(train_b, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    ref_dataset = get_reference_dataset(testset, n_samples=ref_size)
    ref_loader = DataLoader(ref_dataset, batch_size=64, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Shared init
    init_model = SmallViT()
    init_state = copy.deepcopy(init_model.state_dict())

    # Train clients
    print(f"\n  Training Client A ({epochs} epochs)...")
    model_a = SmallViT(); model_a.load_state_dict(copy.deepcopy(init_state))
    train_model(model_a, loader_a, device, epochs=epochs)

    print(f"\n  Training Client B ({epochs} epochs)...")
    model_b = SmallViT(); model_b.load_state_dict(copy.deepcopy(init_state))
    train_model(model_b, loader_b, device, epochs=epochs)

    acc_a = evaluate(model_a, test_loader, device)
    acc_b = evaluate(model_b, test_loader, device)
    best = max(acc_a, acc_b)
    print(f"\n  Client A: {acc_a:.2f}% | Client B: {acc_b:.2f}%")

    # ── Measure Procrustes quality ──
    print("\n  Measuring Procrustes quality per layer...")
    qualities = measure_layer_qualities(model_a, model_b, ref_loader, device)
    for blk, q in qualities.items():
        print(f"    Block {blk}: quality = {q:.4f}"
              f"  {'[AVG]' if q >= threshold else '[DISTILL]'}")

    # ── Method 1: Naive average ──
    print("\n  [1] Naive average...")
    naive_model = naive_average(model_a, model_b)
    acc_naive = evaluate(naive_model, test_loader, device)
    naive_state = naive_model.state_dict()
    print(f"  Naive: {acc_naive:.2f}%")

    # ── Method 2: FedDF warm (baseline from Exp 4) ──
    print("\n  [2] FedDF warm-start...")
    feddf_model = feddf_warm([model_a, model_b], ref_dataset,
                              naive_state, device,
                              distill_epochs=distill_epochs,
                              temperature=temperature)
    acc_feddf = evaluate(feddf_model, test_loader, device)
    print(f"  FedDF warm: {acc_feddf:.2f}%")

    # ── Method 3: FedAD (layer-adaptive, uniform ensemble) ──
    print("\n  [3] FedAD (layer-adaptive hybrid)...")
    fedad_model = fedad_aggregate(
        [model_a, model_b], ref_dataset, qualities, device,
        threshold=threshold, distill_epochs=distill_epochs,
        temperature=temperature, hint_weight=1.0,
        use_confidence_weighting=False)
    acc_fedad = evaluate(fedad_model, test_loader, device)
    print(f"  FedAD: {acc_fedad:.2f}%")

    # ── Method 4: FedAD+CW (hybrid + confidence weighting) ──
    print("\n  [4] FedAD+CW (hybrid + confidence-weighted ensemble)...")
    fedad_cw_model = fedad_aggregate(
        [model_a, model_b], ref_dataset, qualities, device,
        threshold=threshold, distill_epochs=distill_epochs,
        temperature=temperature, hint_weight=1.0,
        use_confidence_weighting=True)
    acc_fedad_cw = evaluate(fedad_cw_model, test_loader, device)
    print(f"  FedAD+CW: {acc_fedad_cw:.2f}%")

    gap = best - acc_naive
    results = {
        'heterogeneity': heterogeneity,
        'local_epochs': epochs,
        'threshold': threshold,
        'client_a': round(acc_a, 2),
        'client_b': round(acc_b, 2),
        'best_individual': round(best, 2),
        'naive_avg': round(acc_naive, 2),
        'feddf_warm': round(acc_feddf, 2),
        'fedad': round(acc_fedad, 2),
        'fedad_cw': round(acc_fedad_cw, 2),
        'gap': round(gap, 2),
        'qualities': {k: round(v, 4) for k, v in qualities.items()},
        'recovery_feddf': round((acc_feddf - acc_naive) / (gap + 1e-10) * 100, 1),
        'recovery_fedad': round((acc_fedad - acc_naive) / (gap + 1e-10) * 100, 1),
        'recovery_fedad_cw': round((acc_fedad_cw - acc_naive) / (gap + 1e-10) * 100, 1),
        'n_distill_blocks': len([q for q in qualities.values() if q < threshold]),
        'n_avg_blocks': len([q for q in qualities.values() if q >= threshold]),
    }
    return results


# ══════════════════════════════════════════════════════════
# 9. REPORTING
# ══════════════════════════════════════════════════════════

def print_summary(all_results):
    print(f"\n{'='*90}")
    print(f"  SUMMARY — Experiment 5: Layer-Adaptive Hybrid Aggregation (FedAD)")
    print(f"{'='*90}")

    for het in ['mild', 'moderate', 'extreme']:
        group = [r for r in all_results if r['heterogeneity'] == het]
        if not group: continue

        print(f"\n  ── {het.upper()} ──")
        print(f"  {'Ep':>3} | {'Best':>6} | {'Naive':>6} | "
              f"{'FedDF':>6} | {'FedAD':>6} | {'AD+CW':>6} | "
              f"{'Avg/Dist':>8} | "
              f"{'FedDF%':>6} | {'FedAD%':>6} | {'CW%':>6}")
        print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*8}-+-"
              f"{'-'*6}-+-{'-'*6}-+-{'-'*6}")

        for r in sorted(group, key=lambda x: x['local_epochs']):
            print(f"  {r['local_epochs']:>3} | "
                  f"{r['best_individual']:>5.1f}% | "
                  f"{r['naive_avg']:>5.1f}% | "
                  f"{r['feddf_warm']:>5.1f}% | "
                  f"{r['fedad']:>5.1f}% | "
                  f"{r['fedad_cw']:>5.1f}% | "
                  f"{r['n_avg_blocks']:>3}/{r['n_distill_blocks']:<3} | "
                  f"{r['recovery_feddf']:>5.0f}% | "
                  f"{r['recovery_fedad']:>5.0f}% | "
                  f"{r['recovery_fedad_cw']:>5.0f}%")

    print(f"\n  Key:")
    print(f"    FedDF  = standard distillation warm-start (Exp 4 baseline)")
    print(f"    FedAD  = layer-adaptive hybrid (weight-avg high-quality, distill low-quality)")
    print(f"    AD+CW  = FedAD + confidence-weighted ensemble")
    print(f"    Avg/Dist = blocks averaged / blocks distilled")
    print(f"    %      = recovery percentage of naive-avg gap")


def save_plot(all_results, path='experiment5_results.pdf'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hets = ['mild', 'moderate', 'extreme']
    fig, axes = plt.subplots(1, len(hets), figsize=(7 * len(hets), 5.5))

    for ax, het in zip(axes, hets):
        group = sorted([r for r in all_results if r['heterogeneity'] == het],
                       key=lambda x: x['local_epochs'])
        if not group: continue

        epochs = [r['local_epochs'] for r in group]
        ax.plot(epochs, [r['best_individual'] for r in group],
                'o-', color='#2196F3', lw=2, ms=7, label='Best Individual')
        ax.plot(epochs, [r['naive_avg'] for r in group],
                's--', color='#FF9800', lw=2, ms=7, label='Naive Average')
        ax.plot(epochs, [r['feddf_warm'] for r in group],
                'D-', color='#9C27B0', lw=2, ms=7, label='FedDF Warm')
        ax.plot(epochs, [r['fedad'] for r in group],
                '^-', color='#4CAF50', lw=2.5, ms=8, label='FedAD (ours)')
        ax.plot(epochs, [r['fedad_cw'] for r in group],
                'p-', color='#E91E63', lw=2.5, ms=8, label='FedAD+CW (ours)')

        ax.fill_between(epochs,
                        [r['naive_avg'] for r in group],
                        [r['best_individual'] for r in group],
                        alpha=0.08, color='#FF9800')

        ax.set_xlabel('Local Training Epochs', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'{het.capitalize()} Heterogeneity', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_xticks(epochs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 5: Layer-Adaptive Hybrid Aggregation (FedAD)\n'
                 'vs Standard FedDF and Naive Averaging',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {path}")


# ══════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    all_results = []

    for het in ['mild', 'moderate', 'extreme']:
        for epochs in [3, 5, 10, 15]:
            r = run_experiment(
                heterogeneity=het,
                epochs=epochs,
                batch_size=128,
                seed=42,
                distill_epochs=20,
                ref_size=500,
                temperature=3.0,
                threshold=0.4,
            )
            all_results.append(r)

    print_summary(all_results)
    save_plot(all_results, 'experiment5_results.pdf')

    with open('experiment5_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  JSON saved: experiment5_results.json")
    print(f"\n  Total wall time: {(time.time() - t0) / 60:.1f} minutes")
