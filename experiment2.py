"""
Experiment 2: Weight-Based Matching (Hungarian Algorithm)
=========================================================
Goal: Show that aligning transformer neurons via weight matching before
      averaging recovers some/all of the accuracy lost by naive FedAvg.

This builds directly on Experiment 1. We add:
  1. Weight-based head matching  (Hungarian algorithm on attention weights)
  2. Weight-based FFN matching   (Hungarian algorithm on FFN neurons)
  3. Activation-based matching   (match using behavior on reference data)

We compare:
  - Individual models (upper bound)
  - Naive average (FedAvg baseline — the problem)
  - Weight-matched average (does geometric alignment help?)
  - Activation-matched average (does functional alignment help more?)
  - Permuted average (lower bound — worst case)

Run:
  pip install torch torchvision matplotlib scipy
  python experiment2.py

  GPU: ~20-25 min  |  CPU: ~2-3 hours

Output:
  - experiment2_results.pdf
  - experiment2_results.json
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
# 1. MODEL (same SmallViT from Experiment 1)
# ══════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


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

    print(f"  Split: {heterogeneity} | "
          f"A: {len(client_a_idx)} | B: {len(client_b_idx)}")
    for name, idx in [("A", client_a_idx), ("B", client_b_idx)]:
        dist = np.bincount(targets[idx], minlength=10)
        print(f"  Client {name} class dist: {dist}")

    return (Subset(trainset, client_a_idx),
            Subset(trainset, client_b_idx),
            testset)


def get_reference_loader(testset, n_samples=500, batch_size=64):
    """
    Small reference dataset for activation-based matching.
    Uses a subset of the test set — in real FL you'd use a small
    public dataset that all parties agree on.
    """
    indices = np.random.choice(len(testset), n_samples, replace=False)
    ref_subset = Subset(testset, indices)
    return DataLoader(ref_subset, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


# ══════════════════════════════════════════════════════════
# 3. TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════

def train_model(model, train_loader, device, epochs=15, lr=1e-3):
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        scheduler.step()
        print(f"    Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")


def evaluate(model, test_loader, device):
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


# ══════════════════════════════════════════════════════════
# 4. NAIVE AVERAGING (baseline from Experiment 1)
# ══════════════════════════════════════════════════════════

def naive_average(model_a, model_b):
    averaged = copy.deepcopy(model_a)
    sa, sb = model_a.state_dict(), model_b.state_dict()
    averaged.load_state_dict({k: (sa[k] + sb[k]) / 2.0 for k in sa})
    return averaged


# ══════════════════════════════════════════════════════════
# 5. WEIGHT-BASED MATCHING
# ══════════════════════════════════════════════════════════
#
# Strategy: For each transformer block, build a cost matrix
# comparing each head (or FFN neuron) in model_ref vs model_b
# based on L2 distance of their weight vectors. Solve with
# Hungarian algorithm. Apply the resulting permutation to
# model_b's weights so they align with model_ref.

def compute_head_cost_matrix(state_ref, state_b, block_idx,
                              embed_dim, num_heads):
    """
    Build an (H x H) cost matrix where C[i,j] = L2 distance between
    head i in reference and head j in model_b.

    Each head's "fingerprint" is the concatenation of its Q, K, V
    projection rows and its O projection columns.
    """
    head_dim = embed_dim // num_heads
    pfx = f"blocks.{block_idx}"

    in_w_ref = state_ref[f"{pfx}.attn.in_proj_weight"]  # (3E, E)
    in_w_b   = state_b[f"{pfx}.attn.in_proj_weight"]
    in_b_ref = state_ref[f"{pfx}.attn.in_proj_bias"]    # (3E,)
    in_b_b   = state_b[f"{pfx}.attn.in_proj_bias"]
    out_w_ref = state_ref[f"{pfx}.attn.out_proj.weight"] # (E, E)
    out_w_b   = state_b[f"{pfx}.attn.out_proj.weight"]

    cost = np.zeros((num_heads, num_heads))

    for i in range(num_heads):
        # Build fingerprint for head i in reference
        fp_ref = []
        for qkv in range(3):
            offset = qkv * embed_dim
            s = slice(offset + i * head_dim, offset + i * head_dim + head_dim)
            fp_ref.append(in_w_ref[s].flatten())
            fp_ref.append(in_b_ref[s])
        # Output proj columns for head i
        s = slice(i * head_dim, i * head_dim + head_dim)
        fp_ref.append(out_w_ref[:, s].flatten())
        fp_ref = torch.cat(fp_ref).cpu().numpy()

        for j in range(num_heads):
            # Build fingerprint for head j in model_b
            fp_b = []
            for qkv in range(3):
                offset = qkv * embed_dim
                s = slice(offset + j * head_dim,
                          offset + j * head_dim + head_dim)
                fp_b.append(in_w_b[s].flatten())
                fp_b.append(in_b_b[s])
            s = slice(j * head_dim, j * head_dim + head_dim)
            fp_b.append(out_w_b[:, s].flatten())
            fp_b = torch.cat(fp_b).cpu().numpy()

            cost[i, j] = np.linalg.norm(fp_ref - fp_b)

    return cost


def compute_ffn_cost_matrix(state_ref, state_b, block_idx):
    """
    Build an (M x M) cost matrix where M = FFN intermediate dim.
    C[i,j] = L2 distance between neuron i in ref and neuron j in B.

    Each neuron's fingerprint = row of up-projection weight + bias +
    column of down-projection weight.
    """
    pfx = f"blocks.{block_idx}"
    up_w_ref = state_ref[f"{pfx}.mlp.0.weight"]   # (M, E)
    up_b_ref = state_ref[f"{pfx}.mlp.0.bias"]      # (M,)
    dn_w_ref = state_ref[f"{pfx}.mlp.3.weight"]   # (E, M)

    up_w_b = state_b[f"{pfx}.mlp.0.weight"]
    up_b_b = state_b[f"{pfx}.mlp.0.bias"]
    dn_w_b = state_b[f"{pfx}.mlp.3.weight"]

    M = up_w_ref.shape[0]
    cost = np.zeros((M, M))

    # Vectorized: build all fingerprints at once
    # ref fingerprint matrix: each row is one neuron's fingerprint
    fp_ref = torch.cat([up_w_ref, up_b_ref.unsqueeze(1),
                        dn_w_ref.t()], dim=1).cpu().numpy()  # (M, E+1+E)
    fp_b = torch.cat([up_w_b, up_b_b.unsqueeze(1),
                      dn_w_b.t()], dim=1).cpu().numpy()

    # Pairwise L2 distances
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    ref_sq = np.sum(fp_ref ** 2, axis=1, keepdims=True)   # (M, 1)
    b_sq = np.sum(fp_b ** 2, axis=1, keepdims=True)       # (M, 1)
    cross = fp_ref @ fp_b.T                                # (M, M)
    dist_sq = ref_sq + b_sq.T - 2 * cross
    dist_sq = np.maximum(dist_sq, 0)  # numerical safety
    cost = np.sqrt(dist_sq)

    return cost


def apply_head_permutation(state, block_idx, perm, embed_dim, num_heads):
    """Apply a head permutation to attention weights in-place."""
    head_dim = embed_dim // num_heads
    pfx = f"blocks.{block_idx}"

    in_w = state[f"{pfx}.attn.in_proj_weight"].clone()
    in_b = state[f"{pfx}.attn.in_proj_bias"].clone()
    out_w = state[f"{pfx}.attn.out_proj.weight"].clone()

    new_in_w, new_in_b = in_w.clone(), in_b.clone()
    new_out_w = out_w.clone()

    for qkv in range(3):
        offset = qkv * embed_dim
        for new_h, old_h in enumerate(perm):
            src = slice(offset + old_h * head_dim,
                        offset + old_h * head_dim + head_dim)
            dst = slice(offset + new_h * head_dim,
                        offset + new_h * head_dim + head_dim)
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
    """Apply an FFN neuron permutation in-place."""
    pfx = f"blocks.{block_idx}"
    perm_idx = torch.LongTensor(perm)
    state[f"{pfx}.mlp.0.weight"] = state[f"{pfx}.mlp.0.weight"][perm_idx]
    state[f"{pfx}.mlp.0.bias"]   = state[f"{pfx}.mlp.0.bias"][perm_idx]
    state[f"{pfx}.mlp.3.weight"] = state[f"{pfx}.mlp.3.weight"][:, perm_idx]


def weight_matched_average(model_ref, model_b):
    """
    Align model_b's neurons to model_ref using weight-space matching,
    then average.

    For each layer:
      1. Build cost matrix from weight L2 distances
      2. Solve assignment with Hungarian algorithm
      3. Permute model_b's weights accordingly
    Then average the aligned weights.
    """
    state_ref = model_ref.state_dict()
    state_b = copy.deepcopy(model_b.state_dict())

    num_blocks = len(model_ref.blocks)
    embed_dim = model_ref.blocks[0].attn.embed_dim
    num_heads = model_ref.blocks[0].attn.num_heads

    total_head_cost = 0
    total_ffn_cost = 0

    for blk in range(num_blocks):
        # ── Match attention heads ──
        head_cost = compute_head_cost_matrix(
            state_ref, state_b, blk, embed_dim, num_heads)
        row_ind, col_ind = linear_sum_assignment(head_cost)
        head_perm = col_ind  # col_ind[i] = which head in B maps to head i
        total_head_cost += head_cost[row_ind, col_ind].sum()
        apply_head_permutation(state_b, blk, head_perm, embed_dim, num_heads)

        # ── Match FFN neurons ──
        ffn_cost = compute_ffn_cost_matrix(state_ref, state_b, blk)
        row_ind, col_ind = linear_sum_assignment(ffn_cost)
        ffn_perm = col_ind
        total_ffn_cost += ffn_cost[row_ind, col_ind].sum()
        apply_ffn_permutation(state_b, blk, ffn_perm)

    print(f"    Weight matching total cost — "
          f"heads: {total_head_cost:.2f}, FFN: {total_ffn_cost:.2f}")

    # Average aligned weights
    aligned_b = copy.deepcopy(model_ref)
    aligned_b.load_state_dict(state_b)
    return naive_average(model_ref, aligned_b)


# ══════════════════════════════════════════════════════════
# 6. ACTIVATION-BASED MATCHING
# ══════════════════════════════════════════════════════════
#
# Instead of comparing raw weights, we run a reference dataset
# through both models and compare what each head/neuron DOES.
# This is more robust because it measures functional similarity.

def collect_head_activations(model, ref_loader, device):
    """
    Run reference data through the model and collect each
    attention head's output activations.

    Returns: dict[block_idx] -> tensor of shape (num_heads, N*T, head_dim)
             where N = num reference samples, T = sequence length
    """
    model.to(device)
    model.eval()

    # Register hooks to capture attention output before projection
    head_acts = {}

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights)
            # attn_output shape: (B, T, embed_dim)
            attn_out = output[0].detach()
            if block_idx not in head_acts:
                head_acts[block_idx] = []
            head_acts[block_idx].append(attn_out)
        return hook_fn

    hooks = []
    for i, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for inputs, _ in ref_loader:
            inputs = inputs.to(device)
            model(inputs)

    for h in hooks:
        h.remove()

    # Reshape: split embed_dim into heads
    num_heads = model.blocks[0].attn.num_heads
    embed_dim = model.blocks[0].attn.embed_dim
    head_dim = embed_dim // num_heads

    result = {}
    for blk_idx, act_list in head_acts.items():
        # Concatenate all batches: (total_tokens, embed_dim)
        all_acts = torch.cat(act_list, dim=0)         # (N, T, E)
        all_acts = all_acts.reshape(-1, embed_dim)     # (N*T, E)
        # Split into heads: (N*T, H, head_dim) -> (H, N*T, head_dim)
        all_acts = all_acts.reshape(-1, num_heads, head_dim)
        all_acts = all_acts.permute(1, 0, 2)           # (H, N*T, D)
        result[blk_idx] = all_acts.cpu()

    return result


def collect_ffn_activations(model, ref_loader, device):
    """
    Collect FFN intermediate activations (after up-proj + GELU).

    Returns: dict[block_idx] -> tensor of shape (M, N*T)
             where M = FFN intermediate dim
    """
    model.to(device)
    model.eval()

    ffn_acts = {}

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            # This hooks the GELU activation (mlp.1)
            # output shape: (B, T, mlp_dim)
            if block_idx not in ffn_acts:
                ffn_acts[block_idx] = []
            ffn_acts[block_idx].append(output.detach())
        return hook_fn

    hooks = []
    for i, block in enumerate(model.blocks):
        # Hook after GELU: mlp is Sequential(Linear, GELU, Dropout, Linear, Dropout)
        h = block.mlp[1].register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for inputs, _ in ref_loader:
            inputs = inputs.to(device)
            model(inputs)

    for h in hooks:
        h.remove()

    result = {}
    for blk_idx, act_list in ffn_acts.items():
        all_acts = torch.cat(act_list, dim=0)   # (N, T, M)
        all_acts = all_acts.reshape(-1, all_acts.shape[-1])  # (N*T, M)
        result[blk_idx] = all_acts.t().cpu()    # (M, N*T)

    return result


def linear_cka(X, Y):
    """
    Linear Centered Kernel Alignment.
    X, Y: (n_samples, dim) numpy arrays.
    Returns: scalar similarity in [0, 1].

    CKA is invariant to orthogonal transformations and isotropic
    scaling, making it more robust than L2 for comparing
    representations that may be rotated relative to each other.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-10
    return hsic_xy / denom


def activation_matched_average(model_ref, model_b, ref_loader, device):
    """
    Align model_b to model_ref using activation-based matching,
    then average.

    For each layer:
      1. Collect head/FFN activations from both models on reference data
      2. Build cost matrix using CKA similarity
      3. Solve assignment with Hungarian algorithm
      4. Permute model_b's weights accordingly

    Key advantage over weight matching: CKA captures functional
    similarity even when weights are in different bases.
    """
    print("    Collecting activations from reference model...")
    head_acts_ref = collect_head_activations(model_ref, ref_loader, device)
    ffn_acts_ref = collect_ffn_activations(model_ref, ref_loader, device)

    print("    Collecting activations from client B...")
    head_acts_b = collect_head_activations(model_b, ref_loader, device)
    ffn_acts_b = collect_ffn_activations(model_b, ref_loader, device)

    state_b = copy.deepcopy(model_b.state_dict())
    num_blocks = len(model_ref.blocks)
    embed_dim = model_ref.blocks[0].attn.embed_dim
    num_heads = model_ref.blocks[0].attn.num_heads

    for blk in range(num_blocks):
        # ── Match attention heads via CKA ──
        H = num_heads
        cost = np.zeros((H, H))
        for i in range(H):
            xi = head_acts_ref[blk][i].numpy()  # (N*T, D)
            for j in range(H):
                xj = head_acts_b[blk][j].numpy()
                # CKA is similarity; we want cost = 1 - similarity
                cost[i, j] = 1.0 - linear_cka(xi, xj)

        row_ind, col_ind = linear_sum_assignment(cost)
        head_perm = col_ind
        matched_cost = cost[row_ind, col_ind].mean()
        print(f"    Block {blk} head match: avg cost = {matched_cost:.4f}")
        apply_head_permutation(state_b, blk, head_perm, embed_dim, num_heads)

        # ── Match FFN neurons via cosine similarity ──
        # CKA per-neuron is expensive for large M; use cosine instead
        ffn_ref = ffn_acts_ref[blk].numpy()  # (M, N*T)
        ffn_b = ffn_acts_b[blk].numpy()

        # Normalize rows for cosine similarity
        ref_norm = ffn_ref / (np.linalg.norm(ffn_ref, axis=1, keepdims=True) + 1e-10)
        b_norm = ffn_b / (np.linalg.norm(ffn_b, axis=1, keepdims=True) + 1e-10)
        cos_sim = ref_norm @ b_norm.T  # (M, M)
        ffn_cost = 1.0 - cos_sim

        row_ind, col_ind = linear_sum_assignment(ffn_cost)
        ffn_perm = col_ind
        apply_ffn_permutation(state_b, blk, ffn_perm)

    print("    Activation matching complete.")

    aligned_b = copy.deepcopy(model_ref)
    aligned_b.load_state_dict(state_b)
    return naive_average(model_ref, aligned_b)


# ══════════════════════════════════════════════════════════
# 7. RANDOM PERMUTATION (worst case baseline)
# ══════════════════════════════════════════════════════════

def permute_model(model, seed=42):
    """Randomly permute heads and FFN neurons (same as Experiment 1)."""
    rng = np.random.RandomState(seed)
    permuted = copy.deepcopy(model)
    state = permuted.state_dict()

    for block_idx in range(len(model.blocks)):
        pfx = f"blocks.{block_idx}"
        embed_dim = model.blocks[block_idx].attn.embed_dim
        num_heads = model.blocks[block_idx].attn.num_heads
        head_dim = embed_dim // num_heads

        head_perm = rng.permutation(num_heads)
        apply_head_permutation(state, block_idx, head_perm,
                               embed_dim, num_heads)

        mlp_dim = state[f"{pfx}.mlp.0.weight"].shape[0]
        ffn_perm = rng.permutation(mlp_dim)
        apply_ffn_permutation(state, block_idx, ffn_perm)

    permuted.load_state_dict(state)
    return permuted


# ══════════════════════════════════════════════════════════
# 8. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════

def run_experiment(heterogeneity='moderate', epochs=15, batch_size=128,
                   seed=42):
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT 2: heterogeneity = {heterogeneity}")
    print(f"{'='*65}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # ── Data ──
    train_a, train_b, testset = get_cifar10_splits(
        heterogeneity=heterogeneity)
    loader_a = DataLoader(train_a, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    loader_b = DataLoader(train_b, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Reference data for activation matching (500 samples from test set)
    ref_loader = get_reference_loader(testset, n_samples=500)

    # ── Shared init ──
    init_model = SmallViT()
    init_state = copy.deepcopy(init_model.state_dict())

    # ── Train ──
    print(f"\n  Training Client A ({epochs} epochs)...")
    model_a = SmallViT()
    model_a.load_state_dict(copy.deepcopy(init_state))
    train_model(model_a, loader_a, device, epochs=epochs)

    print(f"\n  Training Client B ({epochs} epochs)...")
    model_b = SmallViT()
    model_b.load_state_dict(copy.deepcopy(init_state))
    train_model(model_b, loader_b, device, epochs=epochs)

    # ── Evaluate individuals ──
    print("\n  Evaluating individual models...")
    acc_a, _ = evaluate(model_a, test_loader, device)
    acc_b, _ = evaluate(model_b, test_loader, device)
    print(f"  Client A: {acc_a:.2f}%")
    print(f"  Client B: {acc_b:.2f}%")

    # ── Method 1: Naive average ──
    print("\n  [1] Naive average (FedAvg)...")
    avg_naive = naive_average(model_a, model_b)
    acc_naive, _ = evaluate(avg_naive, test_loader, device)
    print(f"  Naive average: {acc_naive:.2f}%")

    # ── Method 2: Weight-matched average ──
    print("\n  [2] Weight-matched average...")
    avg_weight = weight_matched_average(model_a, model_b)
    acc_weight, _ = evaluate(avg_weight, test_loader, device)
    print(f"  Weight-matched average: {acc_weight:.2f}%")

    # ── Method 3: Activation-matched average ──
    print("\n  [3] Activation-matched average...")
    avg_act = activation_matched_average(model_a, model_b,
                                          ref_loader, device)
    acc_act, _ = evaluate(avg_act, test_loader, device)
    print(f"  Activation-matched average: {acc_act:.2f}%")

    # ── Method 4: Permuted average (worst case) ──
    print("\n  [4] Permuted average (worst case)...")
    model_b_perm = permute_model(model_b, seed=123)
    avg_perm = naive_average(model_a, model_b_perm)
    acc_perm, _ = evaluate(avg_perm, test_loader, device)
    print(f"  Permuted average: {acc_perm:.2f}%")

    best = max(acc_a, acc_b)
    results = {
        'heterogeneity':        heterogeneity,
        'client_a_acc':         round(acc_a, 2),
        'client_b_acc':         round(acc_b, 2),
        'best_individual':      round(best, 2),
        'naive_avg_acc':        round(acc_naive, 2),
        'weight_matched_acc':   round(acc_weight, 2),
        'activation_matched_acc': round(acc_act, 2),
        'permuted_avg_acc':     round(acc_perm, 2),
        # Recovery metrics
        'degradation_naive':    round(best - acc_naive, 2),
        'degradation_weight':   round(best - acc_weight, 2),
        'degradation_act':      round(best - acc_act, 2),
        'recovery_weight':      round(acc_weight - acc_naive, 2),
        'recovery_activation':  round(acc_act - acc_naive, 2),
        'recovery_pct_weight':  round(
            (acc_weight - acc_naive) / (best - acc_naive + 1e-10) * 100, 1),
        'recovery_pct_act':     round(
            (acc_act - acc_naive) / (best - acc_naive + 1e-10) * 100, 1),
    }
    return results


# ══════════════════════════════════════════════════════════
# 9. REPORTING
# ══════════════════════════════════════════════════════════

def print_summary(all_results):
    print(f"\n{'='*80}")
    print(f"  SUMMARY — Experiment 2: Does Matching Recover Averaging Performance?")
    print(f"{'='*80}")
    print(f"  Model: SmallViT (4L, 4H, E=128) | Data: CIFAR-10")
    print(f"{'='*80}\n")

    # Table 1: Raw accuracies
    print(f"  {'Het.':<10} | {'Best':>6} | {'Naive':>6} | "
          f"{'W-Match':>7} | {'A-Match':>7} | {'Perm':>6}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*6}-+-"
          f"{'-'*7}-+-{'-'*7}-+-{'-'*6}")
    for r in all_results:
        print(f"  {r['heterogeneity']:<10} | "
              f"{r['best_individual']:>5.1f}% | "
              f"{r['naive_avg_acc']:>5.1f}% | "
              f"{r['weight_matched_acc']:>6.1f}% | "
              f"{r['activation_matched_acc']:>6.1f}% | "
              f"{r['permuted_avg_acc']:>5.1f}%")

    # Table 2: Recovery analysis
    print(f"\n  Recovery Analysis (how much of the naive-avg gap is closed):")
    print(f"  {'Het.':<10} | {'Gap':>7} | {'W-Recov':>8} | {'A-Recov':>8} | "
          f"{'W-%':>5} | {'A-%':>5}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+-{'-'*5}")
    for r in all_results:
        print(f"  {r['heterogeneity']:<10} | "
              f"{r['degradation_naive']:>+6.1f}% | "
              f"{r['recovery_weight']:>+7.1f}% | "
              f"{r['recovery_activation']:>+7.1f}% | "
              f"{r['recovery_pct_weight']:>4.0f}% | "
              f"{r['recovery_pct_act']:>4.0f}%")

    print(f"\n  Key:")
    print(f"    W-Match  = weight-based matching (Hungarian on weight L2)")
    print(f"    A-Match  = activation-based matching (CKA/cosine on ref data)")
    print(f"    Gap      = best individual − naive average")
    print(f"    W/A-%    = % of gap recovered by matching")

    print(f"\n  Interpretation:")
    for r in all_results:
        h = r['heterogeneity']
        wp = r['recovery_pct_weight']
        ap = r['recovery_pct_act']
        if ap > wp + 5:
            verdict = (f"Activation matching ({ap:.0f}%) substantially "
                       f"outperforms weight matching ({wp:.0f}%)")
        elif ap > wp:
            verdict = (f"Activation matching ({ap:.0f}%) slightly "
                       f"outperforms weight matching ({wp:.0f}%)")
        elif wp > ap:
            verdict = (f"Weight matching ({wp:.0f}%) outperforms "
                       f"activation matching ({ap:.0f}%) — unexpected")
        else:
            verdict = f"Both methods recover similarly ({wp:.0f}%)"
        print(f"    {h}: {verdict}")


def save_plot(all_results, path='experiment2_results.pdf'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        labels = ['Best\nIndiv.', 'Naive\nAvg', 'Weight\nMatch',
                  'Activ.\nMatch', 'Permuted\nAvg']
        accs = [r['best_individual'], r['naive_avg_acc'],
                r['weight_matched_acc'], r['activation_matched_acc'],
                r['permuted_avg_acc']]
        colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336']

        bars = ax.bar(labels, accs, color=colors, edgecolor='white',
                      linewidth=1.5, width=0.6)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Heterogeneity: {r["heterogeneity"]}',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(accs) * 1.18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Recovery annotation
        wr = r['recovery_pct_weight']
        ar = r['recovery_pct_act']
        ax.annotate(f'Recovery: W={wr:.0f}%  A={ar:.0f}%',
                    xy=(0.5, 0.02), xycoords='axes fraction',
                    ha='center', fontsize=10, fontweight='bold',
                    color='#4CAF50' if ar > 30 else '#FF9800',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='lightyellow',
                              edgecolor='gray', alpha=0.8))

    fig.suptitle('Experiment 2: Weight vs Activation Matching\n'
                 'for Federated Transformer Averaging',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved → {path}")


# ══════════════════════════════════════════════════════════
# 10. RUN
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t0 = time.time()

    all_results = []
    for het in ['mild', 'moderate', 'extreme']:
        r = run_experiment(heterogeneity=het, epochs=15,
                           batch_size=128, seed=42)
        all_results.append(r)

    print_summary(all_results)
    save_plot(all_results, 'experiment2_results.pdf')

    with open('experiment2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON saved → experiment2_results.json")
    print(f"\n  Total wall time: {(time.time() - t0) / 60:.1f} minutes")
