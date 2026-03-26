"""
Experiment 3: Epoch Sweep + Procrustes Alignment
=================================================
Goal: Find the crossover point where matching stops helping, and test
      whether rotational alignment (Procrustes) extends the useful range.

We sweep local training epochs [1, 2, 3, 5, 10, 15] and measure:
  1. Naive average (FedAvg baseline)
  2. Weight-matched average (Hungarian on weight L2)
  3. Activation-matched average (Hungarian on CKA/cosine)
  4. Procrustes-aligned average (optimal rotation in activation space)

Hypothesis: Matching helps at short training, fails at long training.
Procrustes extends the useful range by handling continuous symmetries.

Run:
  pip install torch torchvision matplotlib scipy
  python experiment3.py

  GPU: ~30-40 min  |  CPU: ~3-4 hours

Output:
  - experiment3_crossover.pdf   (the key plot: recovery vs epochs)
  - experiment3_results.json    (all raw numbers)
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
        x = self.norm(x[:, 0])
        return self.head(x)


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
    n = len(targets)

    if heterogeneity == 'mild':
        indices = np.random.permutation(n)
        split = n // 2
        client_a_idx, client_b_idx = indices[:split], indices[split:]
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
        client_a_idx, client_b_idx = np.array(client_a_idx), np.array(client_b_idx)
    elif heterogeneity == 'extreme':
        client_a_idx = np.where(targets < 5)[0]
        client_b_idx = np.where(targets >= 5)[0]

    train_a = Subset(trainset, client_a_idx)
    train_b = Subset(trainset, client_b_idx)
    return train_a, train_b, testset


def get_reference_loader(testset, n_samples=500, batch_size=64):
    indices = np.random.choice(len(testset), n_samples, replace=False)
    return DataLoader(Subset(testset, indices), batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=True)


# ══════════════════════════════════════════════════════════
# 3. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════

def train_model(model, train_loader, device, epochs=15, lr=1e-3):
    model.to(device)
    model.train()
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
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        scheduler.step()
        if epochs <= 5 or (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Acc: {100.*correct/total:.1f}%")


def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, predicted = model(inputs).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


# ══════════════════════════════════════════════════════════
# 4. AVERAGING METHODS
# ══════════════════════════════════════════════════════════

def naive_average(model_a, model_b):
    averaged = copy.deepcopy(model_a)
    sa, sb = model_a.state_dict(), model_b.state_dict()
    averaged.load_state_dict({k: (sa[k] + sb[k]) / 2.0 for k in sa})
    return averaged


# ── Permutation helpers ──

def apply_head_permutation(state, block_idx, perm, embed_dim, num_heads):
    head_dim = embed_dim // num_heads
    pfx = f"blocks.{block_idx}"
    in_w = state[f"{pfx}.attn.in_proj_weight"].clone()
    in_b = state[f"{pfx}.attn.in_proj_bias"].clone()
    out_w = state[f"{pfx}.attn.out_proj.weight"].clone()
    new_in_w, new_in_b, new_out_w = in_w.clone(), in_b.clone(), out_w.clone()
    for qkv in range(3):
        offset = qkv * embed_dim
        for new_h, old_h in enumerate(perm):
            src = slice(offset + old_h * head_dim, offset + old_h * head_dim + head_dim)
            dst = slice(offset + new_h * head_dim, offset + new_h * head_dim + head_dim)
            new_in_w[dst] = in_w[src]
            new_in_b[dst] = in_b[src]
    for new_h, old_h in enumerate(perm):
        src = slice(old_h * head_dim, old_h * head_dim + head_dim)
        dst = slice(new_h * head_dim, new_h * head_dim + head_dim)
        new_out_w[:, dst] = out_w[:, src]
    state[f"{pfx}.attn.in_proj_weight"] = new_in_w
    state[f"{pfx}.attn.in_proj_bias"] = new_in_b
    state[f"{pfx}.attn.out_proj.weight"] = new_out_w


def apply_ffn_permutation(state, block_idx, perm):
    pfx = f"blocks.{block_idx}"
    perm_idx = torch.LongTensor(perm)
    state[f"{pfx}.mlp.0.weight"] = state[f"{pfx}.mlp.0.weight"][perm_idx]
    state[f"{pfx}.mlp.0.bias"]   = state[f"{pfx}.mlp.0.bias"][perm_idx]
    state[f"{pfx}.mlp.3.weight"] = state[f"{pfx}.mlp.3.weight"][:, perm_idx]


# ── Weight-based matching ──

def weight_matched_average(model_ref, model_b):
    state_ref = model_ref.state_dict()
    state_b = copy.deepcopy(model_b.state_dict())
    num_blocks = len(model_ref.blocks)
    embed_dim = model_ref.blocks[0].attn.embed_dim
    num_heads = model_ref.blocks[0].attn.num_heads
    head_dim = embed_dim // num_heads

    for blk in range(num_blocks):
        pfx = f"blocks.{blk}"

        # Head matching
        H = num_heads
        cost = np.zeros((H, H))
        for i in range(H):
            fp_ref = []
            for qkv in range(3):
                o = qkv * embed_dim
                s = slice(o + i*head_dim, o + i*head_dim + head_dim)
                fp_ref.append(state_ref[f"{pfx}.attn.in_proj_weight"][s].flatten())
                fp_ref.append(state_ref[f"{pfx}.attn.in_proj_bias"][s])
            s = slice(i*head_dim, i*head_dim + head_dim)
            fp_ref.append(state_ref[f"{pfx}.attn.out_proj.weight"][:, s].flatten())
            fp_ref = torch.cat(fp_ref).cpu().numpy()

            for j in range(H):
                fp_b = []
                for qkv in range(3):
                    o = qkv * embed_dim
                    s = slice(o + j*head_dim, o + j*head_dim + head_dim)
                    fp_b.append(state_b[f"{pfx}.attn.in_proj_weight"][s].flatten())
                    fp_b.append(state_b[f"{pfx}.attn.in_proj_bias"][s])
                s = slice(j*head_dim, j*head_dim + head_dim)
                fp_b.append(state_b[f"{pfx}.attn.out_proj.weight"][:, s].flatten())
                fp_b = torch.cat(fp_b).cpu().numpy()
                cost[i, j] = np.linalg.norm(fp_ref - fp_b)

        _, col_ind = linear_sum_assignment(cost)
        apply_head_permutation(state_b, blk, col_ind, embed_dim, num_heads)

        # FFN matching
        up_ref = state_ref[f"{pfx}.mlp.0.weight"]
        ub_ref = state_ref[f"{pfx}.mlp.0.bias"]
        dn_ref = state_ref[f"{pfx}.mlp.3.weight"]
        up_b = state_b[f"{pfx}.mlp.0.weight"]
        ub_b = state_b[f"{pfx}.mlp.0.bias"]
        dn_b = state_b[f"{pfx}.mlp.3.weight"]

        fp_ref = torch.cat([up_ref, ub_ref.unsqueeze(1), dn_ref.t()], dim=1).cpu().numpy()
        fp_b = torch.cat([up_b, ub_b.unsqueeze(1), dn_b.t()], dim=1).cpu().numpy()
        ref_sq = np.sum(fp_ref**2, axis=1, keepdims=True)
        b_sq = np.sum(fp_b**2, axis=1, keepdims=True)
        dist_sq = np.maximum(ref_sq + b_sq.T - 2 * fp_ref @ fp_b.T, 0)
        _, col_ind = linear_sum_assignment(np.sqrt(dist_sq))
        apply_ffn_permutation(state_b, blk, col_ind)

    aligned_b = copy.deepcopy(model_ref)
    aligned_b.load_state_dict(state_b)
    return naive_average(model_ref, aligned_b)


# ── Activation collection ──

def collect_layer_outputs(model, ref_loader, device):
    """
    Collect the FULL residual stream output after each transformer block.
    Shape per block: (N*T, embed_dim)

    This is what Procrustes alignment operates on — the full
    representation, not individual heads.
    """
    model.to(device)
    model.eval()
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

    for h in hooks:
        h.remove()

    result = {}
    for idx, outs in layer_outs.items():
        all_out = torch.cat(outs, dim=0)          # (N, T, E)
        result[idx] = all_out.reshape(-1, all_out.shape[-1]).cpu()  # (N*T, E)
    return result


def collect_head_activations(model, ref_loader, device):
    model.to(device)
    model.eval()
    head_acts = {}

    hooks = []
    for i, block in enumerate(model.blocks):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if idx not in head_acts:
                    head_acts[idx] = []
                head_acts[idx].append(output[0].detach())
            return hook_fn
        hooks.append(block.attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for inputs, _ in ref_loader:
            model(inputs.to(device))

    for h in hooks:
        h.remove()

    num_heads = model.blocks[0].attn.num_heads
    embed_dim = model.blocks[0].attn.embed_dim
    head_dim = embed_dim // num_heads

    result = {}
    for idx, acts in head_acts.items():
        all_acts = torch.cat(acts, dim=0).reshape(-1, embed_dim)
        all_acts = all_acts.reshape(-1, num_heads, head_dim).permute(1, 0, 2)
        result[idx] = all_acts.cpu()
    return result


def collect_ffn_activations(model, ref_loader, device):
    model.to(device)
    model.eval()
    ffn_acts = {}

    hooks = []
    for i, block in enumerate(model.blocks):
        def make_hook(idx):
            def hook_fn(module, input, output):
                if idx not in ffn_acts:
                    ffn_acts[idx] = []
                ffn_acts[idx].append(output.detach())
            return hook_fn
        hooks.append(block.mlp[1].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for inputs, _ in ref_loader:
            model(inputs.to(device))

    for h in hooks:
        h.remove()

    result = {}
    for idx, acts in ffn_acts.items():
        all_acts = torch.cat(acts, dim=0).reshape(-1, acts[0].shape[-1])
        result[idx] = all_acts.t().cpu()
    return result


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


# ── Activation-based matching (same as Exp 2) ──

def activation_matched_average(model_ref, model_b, ref_loader, device):
    head_acts_ref = collect_head_activations(model_ref, ref_loader, device)
    head_acts_b = collect_head_activations(model_b, ref_loader, device)
    ffn_acts_ref = collect_ffn_activations(model_ref, ref_loader, device)
    ffn_acts_b = collect_ffn_activations(model_b, ref_loader, device)

    state_b = copy.deepcopy(model_b.state_dict())
    num_blocks = len(model_ref.blocks)
    embed_dim = model_ref.blocks[0].attn.embed_dim
    num_heads = model_ref.blocks[0].attn.num_heads

    for blk in range(num_blocks):
        H = num_heads
        cost = np.zeros((H, H))
        for i in range(H):
            xi = head_acts_ref[blk][i].numpy()
            for j in range(H):
                xj = head_acts_b[blk][j].numpy()
                cost[i, j] = 1.0 - linear_cka(xi, xj)
        _, col_ind = linear_sum_assignment(cost)
        apply_head_permutation(state_b, blk, col_ind, embed_dim, num_heads)

        ffn_ref = ffn_acts_ref[blk].numpy()
        ffn_b = ffn_acts_b[blk].numpy()
        ref_n = ffn_ref / (np.linalg.norm(ffn_ref, axis=1, keepdims=True) + 1e-10)
        b_n = ffn_b / (np.linalg.norm(ffn_b, axis=1, keepdims=True) + 1e-10)
        _, col_ind = linear_sum_assignment(1.0 - ref_n @ b_n.T)
        apply_ffn_permutation(state_b, blk, col_ind)

    aligned_b = copy.deepcopy(model_ref)
    aligned_b.load_state_dict(state_b)
    return naive_average(model_ref, aligned_b)


# ══════════════════════════════════════════════════════════
# 5. PROCRUSTES ALIGNMENT (the new method)
# ══════════════════════════════════════════════════════════
#
# Instead of just permuting neurons, we find the optimal
# ORTHOGONAL transformation that maps model_b's representations
# to model_ref's representations. This handles continuous
# rotational symmetries that permutation matching can't.
#
# For layer l:
#   Given activations X_ref (N, E) and X_b (N, E),
#   find orthogonal matrix R that minimizes ||X_ref - X_b @ R||
#   Solution: R = V @ U^T  where  U S V^T = SVD(X_ref^T @ X_b)
#
# We then transform model_b's weights:
#   - Output weights of layer l:   W_out  ->  R^T @ W_out
#   - Input weights of layer l+1:  W_in   ->  W_in @ R
#
# This is exact for the linear components. The nonlinearities
# (GELU, softmax) mean it's approximate, but for small rotations
# it works well.

def compute_procrustes_rotation(X_ref, X_b):
    """
    Find orthogonal R minimizing ||X_ref - X_b @ R||_F

    Solution via SVD:  X_ref^T @ X_b = U S V^T  =>  R = V @ U^T

    Returns R as a torch tensor.
    """
    # Center the activations
    X_ref_c = X_ref - X_ref.mean(0)
    X_b_c = X_b - X_b.mean(0)

    # SVD of cross-covariance
    M = X_ref_c.T @ X_b_c  # (E, E)
    U, S, Vt = np.linalg.svd(M, full_matrices=True)

    # Optimal rotation (ensuring proper rotation, det = +1)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0] * (U.shape[1] - 1) + [np.sign(d)])
    R = Vt.T @ D @ U.T  # (E, E)

    # Alignment quality: fraction of variance explained
    aligned = X_b_c @ R
    residual = np.linalg.norm(X_ref_c - aligned, 'fro')
    total = np.linalg.norm(X_ref_c, 'fro')
    quality = 1.0 - (residual / (total + 1e-10))

    return torch.from_numpy(R).float(), quality


def procrustes_aligned_average(model_ref, model_b, ref_loader, device):
    """
    Align model_b to model_ref using Procrustes (optimal rotation)
    on the residual stream, then average.

    For each layer, we:
    1. Collect residual stream activations from both models
    2. Compute the optimal orthogonal transform R
    3. Apply R to model_b's weights:
       - Rotate the output side of the current layer
       - Rotate the input side of the next layer
    4. Also apply permutation matching within heads/FFN
       (Procrustes handles the residual stream rotation;
        permutation handles within-layer reordering)

    This is a TWO-STAGE alignment:
      Stage 1: Procrustes rotation of the residual stream
      Stage 2: Permutation matching within each layer
    """
    print("    Collecting layer outputs for Procrustes...")
    layer_outs_ref = collect_layer_outputs(model_ref, ref_loader, device)
    layer_outs_b = collect_layer_outputs(model_b, ref_loader, device)

    state_b = copy.deepcopy(model_b.state_dict())
    num_blocks = len(model_ref.blocks)
    embed_dim = model_ref.blocks[0].attn.embed_dim
    num_heads = model_ref.blocks[0].attn.num_heads

    # ── Stage 1: Procrustes rotation of residual stream ──
    # We compute the rotation at each layer's output and apply it
    # to transform model_b's weights so its residual stream aligns
    # with model_ref's.

    rotations = []
    for blk in range(num_blocks):
        X_ref = layer_outs_ref[blk].numpy()  # (N*T, E)
        X_b = layer_outs_b[blk].numpy()

        R, quality = compute_procrustes_rotation(X_ref, X_b)
        rotations.append(R)
        print(f"    Block {blk} Procrustes quality: {quality:.4f}")

    # Apply rotations to weights.
    # The residual stream after block l has been rotated by R_l.
    # This means:
    #   - Block l's output projection and LayerNorm need R_l applied
    #   - Block l+1's input projections need R_l applied
    #
    # For attention: out_proj produces the residual addition.
    #   out_proj.weight (E, E): rows are output features -> left-multiply by R^T
    #   So: W_out_new = R^T @ W_out
    #
    # For the NEXT block's attention: in_proj reads from residual stream
    #   in_proj.weight (3E, E): columns are input features -> right-multiply by R
    #   So: W_in_new = W_in @ R
    #
    # Similarly for FFN and LayerNorm.

    # Detect device from the state dict
    weight_device = state_b["blocks.0.attn.out_proj.weight"].device

    for blk in range(num_blocks):
        pfx = f"blocks.{blk}"
        R = rotations[blk].to(weight_device)
        Rt = R.t()

        # Rotate output side of this block's attention
        # out_proj.weight: (E, E) -> R^T @ W
        state_b[f"{pfx}.attn.out_proj.weight"] = Rt @ state_b[f"{pfx}.attn.out_proj.weight"]
        state_b[f"{pfx}.attn.out_proj.bias"] = Rt @ state_b[f"{pfx}.attn.out_proj.bias"]

        # Rotate output side of this block's FFN (down projection)
        # mlp.3.weight: (E, M) -> R^T @ W
        state_b[f"{pfx}.mlp.3.weight"] = Rt @ state_b[f"{pfx}.mlp.3.weight"]
        state_b[f"{pfx}.mlp.3.bias"] = Rt @ state_b[f"{pfx}.mlp.3.bias"]

        # Rotate input side of NEXT block (or final norm if last block)
        if blk + 1 < num_blocks:
            npfx = f"blocks.{blk+1}"
            # Next block's norm1 (before attention)
            state_b[f"{npfx}.norm1.weight"] = state_b[f"{npfx}.norm1.weight"] @ R
            # LayerNorm is elementwise scale+shift, so we need to
            # rotate the input. But LN normalizes first, so we actually
            # need to be more careful. For a first approximation,
            # we transform the attention input projections.

            # Next block's attention in_proj: (3E, E) -> W @ R
            state_b[f"{npfx}.attn.in_proj_weight"] = \
                state_b[f"{npfx}.attn.in_proj_weight"] @ R

            # Next block's norm2 (before FFN)
            state_b[f"{npfx}.norm2.weight"] = state_b[f"{npfx}.norm2.weight"] @ R

            # Next block's FFN up-projection: (M, E) -> W @ R
            state_b[f"{npfx}.mlp.0.weight"] = state_b[f"{npfx}.mlp.0.weight"] @ R
        else:
            # Final layer norm
            state_b["norm.weight"] = state_b["norm.weight"] @ R
            # Classification head: (num_classes, E) -> W @ R
            state_b["head.weight"] = state_b["head.weight"] @ R

    # ── Stage 2: Permutation matching within layers ──
    # After rotation, do standard permutation matching for heads and FFN
    state_ref = model_ref.state_dict()

    for blk in range(num_blocks):
        pfx = f"blocks.{blk}"
        H = num_heads
        head_dim = embed_dim // num_heads

        # Head matching on rotated weights
        cost = np.zeros((H, H))
        for i in range(H):
            fp_ref = []
            for qkv in range(3):
                o = qkv * embed_dim
                s = slice(o + i*head_dim, o + i*head_dim + head_dim)
                fp_ref.append(state_ref[f"{pfx}.attn.in_proj_weight"][s].flatten())
                fp_ref.append(state_ref[f"{pfx}.attn.in_proj_bias"][s])
            s = slice(i*head_dim, i*head_dim + head_dim)
            fp_ref.append(state_ref[f"{pfx}.attn.out_proj.weight"][:, s].flatten())
            fp_ref = torch.cat(fp_ref).cpu().numpy()

            for j in range(H):
                fp_b = []
                for qkv in range(3):
                    o = qkv * embed_dim
                    s = slice(o + j*head_dim, o + j*head_dim + head_dim)
                    fp_b.append(state_b[f"{pfx}.attn.in_proj_weight"][s].flatten())
                    fp_b.append(state_b[f"{pfx}.attn.in_proj_bias"][s])
                s = slice(j*head_dim, j*head_dim + head_dim)
                fp_b.append(state_b[f"{pfx}.attn.out_proj.weight"][:, s].flatten())
                fp_b = torch.cat(fp_b).cpu().numpy()
                cost[i, j] = np.linalg.norm(fp_ref - fp_b)

        _, col_ind = linear_sum_assignment(cost)
        apply_head_permutation(state_b, blk, col_ind, embed_dim, num_heads)

        # FFN matching
        up_ref = state_ref[f"{pfx}.mlp.0.weight"]
        ub_ref = state_ref[f"{pfx}.mlp.0.bias"]
        dn_ref = state_ref[f"{pfx}.mlp.3.weight"]
        up_b = state_b[f"{pfx}.mlp.0.weight"]
        ub_b = state_b[f"{pfx}.mlp.0.bias"]
        dn_b = state_b[f"{pfx}.mlp.3.weight"]

        fp_r = torch.cat([up_ref, ub_ref.unsqueeze(1), dn_ref.t()], dim=1).cpu().numpy()
        fp_b = torch.cat([up_b, ub_b.unsqueeze(1), dn_b.t()], dim=1).cpu().numpy()
        r_sq = np.sum(fp_r**2, axis=1, keepdims=True)
        b_sq = np.sum(fp_b**2, axis=1, keepdims=True)
        dist_sq = np.maximum(r_sq + b_sq.T - 2 * fp_r @ fp_b.T, 0)
        _, col_ind = linear_sum_assignment(np.sqrt(dist_sq))
        apply_ffn_permutation(state_b, blk, col_ind)

    aligned_b = copy.deepcopy(model_ref)
    aligned_b.load_state_dict(state_b)
    return naive_average(model_ref, aligned_b)


# ══════════════════════════════════════════════════════════
# 6. EPOCH SWEEP
# ══════════════════════════════════════════════════════════

def run_epoch_sweep(heterogeneity='moderate', epoch_list=None,
                    batch_size=128, seed=42):
    """
    Train two clients for varying numbers of epochs and measure
    how well each matching method recovers averaging performance.
    """
    if epoch_list is None:
        epoch_list = [1, 2, 3, 5, 10, 15]

    print(f"\n{'#'*65}")
    print(f"  EPOCH SWEEP: heterogeneity = {heterogeneity}")
    print(f"  Epochs to test: {epoch_list}")
    print(f"{'#'*65}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load data once
    train_a, train_b, testset = get_cifar10_splits(
        heterogeneity=heterogeneity)
    loader_a = DataLoader(train_a, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    loader_b = DataLoader(train_b, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    ref_loader = get_reference_loader(testset, n_samples=500)

    # Shared init (reused for all epoch counts)
    init_model = SmallViT()
    init_state = copy.deepcopy(init_model.state_dict())

    sweep_results = []

    for epochs in epoch_list:
        print(f"\n  {'─'*55}")
        print(f"  Training with {epochs} local epoch(s)...")
        print(f"  {'─'*55}")

        # Reset seeds for reproducibility within each run
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train Client A
        model_a = SmallViT()
        model_a.load_state_dict(copy.deepcopy(init_state))
        train_model(model_a, loader_a, device, epochs=epochs)

        # Train Client B
        model_b = SmallViT()
        model_b.load_state_dict(copy.deepcopy(init_state))
        train_model(model_b, loader_b, device, epochs=epochs)

        # Evaluate individuals
        acc_a = evaluate(model_a, test_loader, device)
        acc_b = evaluate(model_b, test_loader, device)
        best = max(acc_a, acc_b)
        print(f"  Client A: {acc_a:.2f}% | Client B: {acc_b:.2f}%")

        # Naive average
        acc_naive = evaluate(naive_average(model_a, model_b),
                             test_loader, device)
        print(f"  Naive avg: {acc_naive:.2f}%")

        # Weight-matched
        acc_weight = evaluate(weight_matched_average(model_a, model_b),
                              test_loader, device)
        print(f"  Weight-matched: {acc_weight:.2f}%")

        # Activation-matched
        acc_act = evaluate(
            activation_matched_average(model_a, model_b, ref_loader, device),
            test_loader, device)
        print(f"  Activation-matched: {acc_act:.2f}%")

        # Procrustes-aligned
        acc_proc = evaluate(
            procrustes_aligned_average(model_ref=model_a, model_b=model_b,
                                       ref_loader=ref_loader, device=device),
            test_loader, device)
        print(f"  Procrustes-aligned: {acc_proc:.2f}%")

        gap = best - acc_naive
        sweep_results.append({
            'epochs': epochs,
            'client_a': round(acc_a, 2),
            'client_b': round(acc_b, 2),
            'best': round(best, 2),
            'naive': round(acc_naive, 2),
            'weight_matched': round(acc_weight, 2),
            'activation_matched': round(acc_act, 2),
            'procrustes': round(acc_proc, 2),
            'gap': round(gap, 2),
            'recovery_weight': round(acc_weight - acc_naive, 2),
            'recovery_act': round(acc_act - acc_naive, 2),
            'recovery_proc': round(acc_proc - acc_naive, 2),
            'recovery_pct_weight': round(
                (acc_weight - acc_naive) / (gap + 1e-10) * 100, 1),
            'recovery_pct_act': round(
                (acc_act - acc_naive) / (gap + 1e-10) * 100, 1),
            'recovery_pct_proc': round(
                (acc_proc - acc_naive) / (gap + 1e-10) * 100, 1),
        })

    return {'heterogeneity': heterogeneity, 'sweeps': sweep_results}


# ══════════════════════════════════════════════════════════
# 7. REPORTING
# ══════════════════════════════════════════════════════════

def print_summary(all_sweeps):
    for sweep in all_sweeps:
        het = sweep['heterogeneity']
        results = sweep['sweeps']

        print(f"\n{'='*80}")
        print(f"  {het.upper()} HETEROGENEITY — Epoch Sweep")
        print(f"{'='*80}")
        print(f"  {'Ep':>3} | {'Best':>6} | {'Naive':>6} | {'W-Mat':>6} | "
              f"{'A-Mat':>6} | {'Procr':>6} | {'Gap':>6} | "
              f"{'W-%':>5} | {'A-%':>5} | {'P-%':>5}")
        print(f"  {'-'*3}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*5}-+-{'-'*5}-+-{'-'*5}")

        for r in results:
            print(f"  {r['epochs']:>3} | "
                  f"{r['best']:>5.1f}% | {r['naive']:>5.1f}% | "
                  f"{r['weight_matched']:>5.1f}% | "
                  f"{r['activation_matched']:>5.1f}% | "
                  f"{r['procrustes']:>5.1f}% | "
                  f"{r['gap']:>+5.1f}% | "
                  f"{r['recovery_pct_weight']:>4.0f}% | "
                  f"{r['recovery_pct_act']:>4.0f}% | "
                  f"{r['recovery_pct_proc']:>4.0f}%")

    print(f"\n  Key:")
    print(f"    W-Mat  = weight-matched average")
    print(f"    A-Mat  = activation-matched average (CKA/cosine)")
    print(f"    Procr  = Procrustes rotation + permutation matching")
    print(f"    Gap    = best individual − naive average")
    print(f"    W/A/P-% = percentage of gap recovered")


def save_plot(all_sweeps, path='experiment3_crossover.pdf'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(all_sweeps)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, sweep in zip(axes, all_sweeps):
        het = sweep['heterogeneity']
        results = sweep['sweeps']

        epochs = [r['epochs'] for r in results]
        best_acc = [r['best'] for r in results]
        naive_acc = [r['naive'] for r in results]
        weight_acc = [r['weight_matched'] for r in results]
        act_acc = [r['activation_matched'] for r in results]
        proc_acc = [r['procrustes'] for r in results]

        ax.plot(epochs, best_acc, 'o-', color='#2196F3', linewidth=2,
                markersize=7, label='Best Individual', zorder=5)
        ax.plot(epochs, naive_acc, 's--', color='#FF9800', linewidth=2,
                markersize=7, label='Naive Average', zorder=4)
        ax.plot(epochs, weight_acc, '^-', color='#9C27B0', linewidth=2,
                markersize=7, label='Weight-Matched', zorder=3)
        ax.plot(epochs, act_acc, 'D-', color='#E91E63', linewidth=2,
                markersize=7, label='Activation-Matched', zorder=3)
        ax.plot(epochs, proc_acc, 'p-', color='#4CAF50', linewidth=2,
                markersize=8, label='Procrustes', zorder=5)

        # Shade the gap
        ax.fill_between(epochs, naive_acc, best_acc,
                         alpha=0.1, color='#FF9800', label='_nolegend_')

        ax.set_xlabel('Local Training Epochs', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Heterogeneity: {het}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks(epochs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: Recovery vs Local Training Duration\n'
                 '(Does matching help more at shorter training?)',
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → {path}")


def save_recovery_plot(all_sweeps, path='experiment3_recovery.pdf'):
    """Plot recovery percentage vs epochs — the clearest view."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(all_sweeps)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, sweep in zip(axes, all_sweeps):
        het = sweep['heterogeneity']
        results = sweep['sweeps']

        epochs = [r['epochs'] for r in results]
        w_pct = [r['recovery_pct_weight'] for r in results]
        a_pct = [r['recovery_pct_act'] for r in results]
        p_pct = [r['recovery_pct_proc'] for r in results]

        ax.plot(epochs, w_pct, '^-', color='#9C27B0', linewidth=2,
                markersize=7, label='Weight-Matched')
        ax.plot(epochs, a_pct, 'D-', color='#E91E63', linewidth=2,
                markersize=7, label='Activation-Matched')
        ax.plot(epochs, p_pct, 'p-', color='#4CAF50', linewidth=2,
                markersize=8, label='Procrustes')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Local Training Epochs', fontsize=12)
        ax.set_ylabel('Recovery (%)', fontsize=12)
        ax.set_title(f'Heterogeneity: {het}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.set_xticks(epochs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 3: Gap Recovery (%) vs Local Training Epochs\n'
                 '(How much of the naive-averaging damage is repaired?)',
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

    # Run sweep for each heterogeneity level
    # Start with moderate (most realistic), then mild and extreme
    for het in ['mild', 'moderate', 'extreme']:
        sweep = run_epoch_sweep(
            heterogeneity=het,
            epoch_list=[1, 2, 3, 5, 10, 15],
            batch_size=128,
            seed=42,
        )
        all_sweeps.append(sweep)

    print_summary(all_sweeps)
    save_plot(all_sweeps, 'experiment3_crossover.pdf')
    save_recovery_plot(all_sweeps, 'experiment3_recovery.pdf')

    with open('experiment3_results.json', 'w') as f:
        json.dump(all_sweeps, f, indent=2)
    print(f"  JSON saved → experiment3_results.json")
    print(f"\n  Total wall time: {(time.time() - t0) / 60:.1f} minutes")
