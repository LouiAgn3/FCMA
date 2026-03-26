# ==============================================================================
# FCMA_CIFAR10.py
# Fed-CMA experiment on CIFAR-10 — the standard FL benchmark
#
# Model: Simple CNN (Conv-Conv-Pool-Conv-Conv-Pool-FC-FC-Out)
# Matching: Channel-wise FedMA for conv layers + neuron-wise for FC layers
#           (FedMA paper Section 2.2, Equation 5)
# Partitioning: Dirichlet(alpha) for controllable non-IID
# Context: Derived from class distribution profile of each client
#
# Supports all three contributions:
#   C1: --agg_mode fedma|fedavg  (matched vs naive within clusters)
#   C2: --init_mode context|random|model  (cold-start comparison)
#   C3: --schedule fixed|linear|cosine  (weight scheduling)
#
# Usage:
#   python FCMA_CIFAR10.py --clients 20 --alpha 0.5
#   python FCMA_CIFAR10.py --clients 20 --alpha 0.1 --mode FedAvg
#   python FCMA_CIFAR10.py --clients 20 --alpha 0.5 --agg_mode fedavg  # C1 ablation
#   python FCMA_CIFAR10.py --clients 20 --alpha 0.5 --init_mode random  # C2 ablation
#   python FCMA_CIFAR10.py --clients 20 --alpha 0.5 --schedule linear  # C3 test
# ==============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import random
import copy
import os
import gc
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

print("Libraries imported successfully.")

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
NUM_ROUNDS = 200
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SEED = 42

GRADIENT_CLIP_NORM = 1.0
MIN_LR = 0.0001

RECLUSTERING_INTERVAL = 40
LOW_RANK_DIM = 10
SIMILARITY_THRESHOLD = 0.1

# Similarity weights
ALPHA = 0.25
BETA = 0.45
GAMMA = 0.30

MODEL_BLEND_WEIGHT = 0.7  # Keep 70% of local model, take 30% from new cluster

NUM_CLASSES = 10
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def get_num_clusters(num_clients):
    return max(2, min(10, int(np.sqrt(num_clients))))


# ==============================================================================
# --- MODEL (CNN for CIFAR-10) ---
# ==============================================================================
class CIFAR_CNN(nn.Module):
    """
    Simple CNN for CIFAR-10. Architecture chosen to be comparable to VGG-9
    from the FedMA paper but smaller for faster experimentation.

    Conv layers have channel-wise permutation invariance (Equation 5).
    FC layers have neuron-wise permutation invariance (Equation 4).

    Architecture:
      conv1: 3 -> 32, 3x3, pad=1
      conv2: 32 -> 64, 3x3, pad=1, then MaxPool 2x2
      conv3: 64 -> 128, 3x3, pad=1
      conv4: 128 -> 128, 3x3, pad=1, then MaxPool 2x2
      fc1: 128*8*8 -> 256
      fc2: 256 -> 128
      fc_out: 128 -> 10
    """
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Conv block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # FC block
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, NUM_CLASSES)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.pool2(torch.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc_out(x)


# ==============================================================================
# --- DATA LOADING & PARTITIONING ---
# ==============================================================================
def load_cifar10():
    """Load CIFAR-10 with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset


def partition_dirichlet(train_dataset, num_clients, alpha=0.5):
    """
    Dirichlet non-IID partitioning — the FL standard.
    Returns: list of client Subset datasets, list of client metadata dicts
    """
    rng = np.random.default_rng(SEED)
    targets = np.array(train_dataset.targets)
    num_classes = NUM_CLASSES

    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        class_indices = np.where(targets == k)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        counts = (proportions * len(class_indices)).astype(int)
        remainder = len(class_indices) - counts.sum()
        if remainder > 0:
            bonus = rng.choice(num_clients, size=remainder, replace=False)
            counts[bonus] += 1
        elif remainder < 0:
            for _ in range(-remainder):
                counts[np.argmax(counts)] -= 1
        offset = 0
        for cid in range(num_clients):
            client_indices[cid].extend(class_indices[offset:offset + counts[cid]].tolist())
            offset += counts[cid]

    # Build subsets and metadata
    client_datasets = []
    client_metadata = []
    for cid in range(num_clients):
        indices = client_indices[cid]
        client_datasets.append(Subset(train_dataset, indices))

        # Derive context metadata from class distribution
        labels = targets[indices]
        class_dist = np.zeros(num_classes)
        for k in range(num_classes):
            class_dist[k] = np.sum(labels == k) / len(labels) if len(labels) > 0 else 0

        # Dominant classes (top 3)
        top_classes = set(np.argsort(class_dist)[-3:].tolist())
        # Data volume tier
        if len(indices) > 3000:
            volume_tier = 'large'
        elif len(indices) > 1000:
            volume_tier = 'medium'
        else:
            volume_tier = 'small'
        # Class concentration (entropy-based)
        entropy = -np.sum(class_dist[class_dist > 0] * np.log(class_dist[class_dist > 0] + 1e-10))
        max_entropy = np.log(num_classes)
        concentration = 'concentrated' if entropy < 0.5 * max_entropy else 'spread'

        client_metadata.append({
            'client_id': cid,
            'class_distribution': class_dist,
            'dominant_classes': top_classes,
            'volume_tier': volume_tier,
            'concentration': concentration,
            'is_poisoned': False,
        })

    return client_datasets, client_metadata


# ==============================================================================
# --- CONTEXT SIMILARITY ---
# ==============================================================================
def calculate_s_context(client_metadata):
    """
    Context similarity for CIFAR-10 Dirichlet-partitioned clients.
    Uses continuous class distribution cosine similarity (much more discriminative
    than coarse Jaccard over top-3 classes).
    """
    n = len(client_metadata)
    s = np.zeros((n, n))
    # Stack all class distributions into a matrix
    dists = np.array([m['class_distribution'] for m in client_metadata])
    # Cosine similarity between distribution vectors
    norms = np.linalg.norm(dists, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normed = dists / norms
    s = normed @ normed.T
    np.clip(s, 0, 1, out=s)
    return s


def calculate_s_data(client_metadata):
    """Data distribution similarity using Jensen-Shannon divergence."""
    n = len(client_metadata)
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            di = client_metadata[i]['class_distribution']
            dj = client_metadata[j]['class_distribution']
            sim = 1 - jensenshannon(di + 1e-10, dj + 1e-10)
            s[i, j] = s[j, i] = sim
    return s


# ==============================================================================
# --- HELPERS ---
# ==============================================================================
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def calculate_model_size(model):
    p = sum(param.nelement() * param.element_size() for param in model.parameters())
    b = sum(buf.nelement() * buf.element_size() for buf in model.buffers())
    return (p + b) / 1024**2


def calculate_s_model(model_updates, M):
    proj = model_updates @ M
    norm = np.linalg.norm(proj, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    cosine_sim = (proj @ proj.T) / (norm @ norm.T)
    return np.clip(cosine_sim, 0, 1)


def evaluate(model, test_loader):
    """Evaluate accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total if total > 0 else 0.0


def evaluate_per_client(model, client_dataset):
    """Evaluate on a client's local data (Subset)."""
    loader = DataLoader(client_dataset, batch_size=256, shuffle=False)
    return evaluate(model, loader)


# ==============================================================================
# --- WEIGHT SCHEDULING (C3) ---
# ==============================================================================
def get_scheduled_weights(round_num, total_rounds, schedule='fixed'):
    """
    Returns (alpha, beta, gamma) for the given round.
    - fixed: constant weights
    - linear: linearly interpolate from context-heavy to model-heavy
    - cosine: cosine annealing from context-heavy to model-heavy
    """
    if schedule == 'fixed':
        return ALPHA, BETA, GAMMA

    # Start: context-heavy (early rounds, model signals are noise)
    a_start, b_start, g_start = 0.10, 0.30, 0.60
    # End: model-heavy (late rounds, models have diverged meaningfully)
    a_end, b_end, g_end = 0.50, 0.40, 0.10

    progress = round_num / max(total_rounds - 1, 1)

    if schedule == 'linear':
        t = progress
    elif schedule == 'cosine':
        t = 0.5 * (1 - np.cos(np.pi * progress))
    else:
        return ALPHA, BETA, GAMMA

    a = a_start + t * (a_end - a_start)
    b = b_start + t * (b_end - b_start)
    g = g_start + t * (g_end - g_start)
    return a, b, g


# ==============================================================================
# --- AGGREGATION ---
# ==============================================================================
def federated_averaging(models):
    """Standard FedAvg."""
    if not models:
        return None
    avg_sd = copy.deepcopy(models[0].state_dict())
    for key in avg_sd:
        avg_sd[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(0)
    agg = CIFAR_CNN().to(DEVICE)
    agg.load_state_dict(avg_sd)
    return agg


def intra_cluster_fedma_cnn(cluster_models, ref_model, threshold):
    """
    FedMA for CNN + FC architecture (FedMA paper Sections 2.2).

    Conv layers: match OUTPUT CHANNELS. Each channel's filter is flattened
    (C_in * w * h) into a single vector for cosine similarity matching.
    The permutation of output channels at layer L propagates to input
    channels at layer L+1 (Equation 5).

    FC layers: standard neuron matching with input dim permutation from
    the last conv layer's channel permutation (flattened).

    Architecture: conv1 -> conv2 -> conv3 -> conv4 -> fc1 -> fc2 -> fc_out
    """
    if not cluster_models:
        return None
    if len(cluster_models) == 1:
        return copy.deepcopy(cluster_models[0])

    ref_sd = ref_model.state_dict()
    n_models = len(cluster_models)
    all_sds = [m.state_dict() for m in cluster_models]
    agg_sd = {}

    # ================================================================
    # PHASE 1: Match conv layers (channel-wise, Equation 5)
    # ================================================================
    conv_layers = [
        {'name': 'conv1', 'out_ch': 32, 'in_ch': 3},
        {'name': 'conv2', 'out_ch': 64, 'in_ch': 32},
        {'name': 'conv3', 'out_ch': 128, 'in_ch': 64},
        {'name': 'conv4', 'out_ch': 128, 'in_ch': 128},
    ]

    prev_perms = [None] * n_models  # No input permutation for first layer (3 RGB channels)

    for layer_info in conv_layers:
        name = layer_info['name']
        out_ch = layer_info['out_ch']
        # weight shape: (out_channels, in_channels, kH, kW)
        ref_w = ref_sd[f'{name}.weight']  # (out_ch, in_ch, 3, 3)

        # Flatten each output channel: (out_ch, in_ch*3*3)
        ref_flat = ref_w.view(out_ch, -1)

        current_perms = []
        for m_idx in range(n_models):
            client_w = all_sds[m_idx][f'{name}.weight'].clone()

            # Apply previous layer's channel permutation to INPUT channels
            if prev_perms[m_idx] is not None:
                inv_prev = np.argsort(prev_perms[m_idx])
                # Permute the input channel dimension (dim 1)
                client_w = client_w[:, inv_prev, :, :]

            client_flat = client_w.view(out_ch, -1)

            # Match output channels using cosine similarity
            cost = 1 - torch.nn.functional.cosine_similarity(
                ref_flat.unsqueeze(1), client_flat.unsqueeze(0), dim=2
            )
            _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            current_perms.append(col_ind)

        # Aggregate with permutations
        sum_w = torch.zeros_like(ref_sd[f'{name}.weight']).float()
        sum_b = torch.zeros_like(ref_sd[f'{name}.bias']).float()

        for m_idx in range(n_models):
            inv_perm = np.argsort(current_perms[m_idx])
            sd = all_sds[m_idx]

            w = sd[f'{name}.weight'].clone()
            # Permute input channels from previous layer
            if prev_perms[m_idx] is not None:
                inv_prev = np.argsort(prev_perms[m_idx])
                w = w[:, inv_prev, :, :]
            # Permute output channels
            w = w[inv_perm, :, :, :]
            sum_w += w

            sum_b += sd[f'{name}.bias'][inv_perm]

        agg_sd[f'{name}.weight'] = sum_w / n_models
        agg_sd[f'{name}.bias'] = sum_b / n_models
        prev_perms = current_perms

    # ================================================================
    # PHASE 2: Match FC layers (neuron-wise, Equation 4)
    # The first FC layer's input dimension comes from the last conv
    # layer's output channels, flattened over spatial dims (8*8).
    # ================================================================

    # Expand conv4 channel permutation to fc1 input dimension
    # fc1 input = 128 channels * 8 * 8 = 8192
    # Channel i of conv4 maps to fc1 input indices [i*64 : (i+1)*64]
    spatial_size = 8 * 8  # After two 2x2 max pools on 32x32

    fc_layers = [
        {'name': 'fc1', 'out_dim': 256, 'in_dim': 128 * spatial_size, 'from_conv': True},
        {'name': 'fc2', 'out_dim': 128, 'in_dim': 256, 'from_conv': False},
    ]

    for layer_info in fc_layers:
        name = layer_info['name']
        ref_w = ref_sd[f'{name}.weight']  # (out_dim, in_dim)

        current_fc_perms = []
        for m_idx in range(n_models):
            client_w = all_sds[m_idx][f'{name}.weight'].clone()

            # Apply previous permutation to input dimension
            if layer_info['from_conv']:
                # Expand channel permutation to flattened spatial dims
                if prev_perms[m_idx] is not None:
                    inv_prev = np.argsort(prev_perms[m_idx])
                    # Build expanded permutation: channel i -> spatial block i*64:(i+1)*64
                    expanded = []
                    for ch in inv_prev:
                        expanded.extend(range(ch * spatial_size, (ch + 1) * spatial_size))
                    client_w = client_w[:, expanded]
            else:
                if prev_perms[m_idx] is not None:
                    inv_prev = np.argsort(prev_perms[m_idx])
                    client_w = client_w[:, inv_prev]

            cost = 1 - torch.nn.functional.cosine_similarity(
                ref_w.unsqueeze(1), client_w.unsqueeze(0), dim=2
            )
            _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            current_fc_perms.append(col_ind)

        # Aggregate
        sum_w = torch.zeros_like(ref_sd[f'{name}.weight']).float()
        sum_b = torch.zeros_like(ref_sd[f'{name}.bias']).float()

        for m_idx in range(n_models):
            inv_perm = np.argsort(current_fc_perms[m_idx])
            sd = all_sds[m_idx]
            w = sd[f'{name}.weight'].clone()

            if layer_info['from_conv']:
                if prev_perms[m_idx] is not None:
                    inv_prev = np.argsort(prev_perms[m_idx])
                    expanded = []
                    for ch in inv_prev:
                        expanded.extend(range(ch * spatial_size, (ch + 1) * spatial_size))
                    w = w[:, expanded]
            else:
                if prev_perms[m_idx] is not None:
                    inv_prev = np.argsort(prev_perms[m_idx])
                    w = w[:, inv_prev]

            w = w[inv_perm, :]
            sum_w += w
            sum_b += sd[f'{name}.bias'][inv_perm]

        agg_sd[f'{name}.weight'] = sum_w / n_models
        agg_sd[f'{name}.bias'] = sum_b / n_models
        prev_perms = current_fc_perms

    # Output layer: permute input dim, average output
    sum_w = torch.zeros_like(ref_sd['fc_out.weight']).float()
    sum_b = torch.zeros_like(ref_sd['fc_out.bias']).float()
    for m_idx in range(n_models):
        inv_prev = np.argsort(prev_perms[m_idx])
        sum_w += all_sds[m_idx]['fc_out.weight'][:, inv_prev]
        sum_b += all_sds[m_idx]['fc_out.bias']
    agg_sd['fc_out.weight'] = sum_w / n_models
    agg_sd['fc_out.bias'] = sum_b / n_models

    agg = CIFAR_CNN().to(DEVICE)
    agg.load_state_dict(agg_sd)
    return agg


# ==============================================================================
# --- MAIN EXPERIMENT ---
# ==============================================================================
def run_experiment(mode, num_clients, alpha, agg_mode, init_mode, schedule):
    num_clusters = get_num_clusters(num_clients)

    run_label = f"{mode}_N{num_clients}_a{alpha}_{agg_mode}_{init_mode}_{schedule}"

    print(f"\n{'='*60}")
    print(f"  {run_label}  |  Clusters: {num_clusters}")
    if mode == 'FCMA':
        print(f"  Agg: {agg_mode} | Init: {init_mode} | Schedule: {schedule}")
    print(f"  Dirichlet alpha: {alpha}")
    print(f"{'='*60}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results_{run_label}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)

    # --- Load data ---
    train_dataset, test_dataset = load_cifar10()
    global_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # --- Partition ---
    client_datasets, client_metadata = partition_dirichlet(train_dataset, num_clients, alpha)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]

    # --- Print distribution ---
    print(f"\n--- Client Distribution (Dirichlet alpha={alpha}) ---")
    for cid in range(min(num_clients, 15)):
        m = client_metadata[cid]
        top = sorted(m['dominant_classes'])
        top_names = [CIFAR_CLASSES[c] for c in top]
        print(f"  Client {cid:2d}: {len(client_datasets[cid]):5d} samples | "
              f"Vol={m['volume_tier']:<6s} Conc={m['concentration']:<12s} "
              f"Top: {', '.join(top_names)}")
    if num_clients > 15:
        print(f"  ... ({num_clients - 15} more)")

    # --- Model init ---
    criterion = nn.CrossEntropyLoss()

    if mode == 'FedAvg':
        global_model = CIFAR_CNN().to(DEVICE)
        client_cluster_assignments = np.zeros(num_clients, dtype=int)
    else:  # FCMA
        cluster_models = [CIFAR_CNN().to(DEVICE) for _ in range(num_clusters)]

        # Initial clustering based on init_mode
        if init_mode == 'context':
            s_data = calculate_s_data(client_metadata)
            s_ctx = calculate_s_context(client_metadata)
            init_sim = 0.4 * s_data + 0.6 * s_ctx
        elif init_mode == 'random':
            init_sim = np.random.rand(num_clients, num_clients)
            init_sim = (init_sim + init_sim.T) / 2
            np.fill_diagonal(init_sim, 1.0)
        else:  # model — use data similarity only (no context, model signals unavailable at round 0)
            init_sim = calculate_s_data(client_metadata)

        client_cluster_assignments = AgglomerativeClustering(
            n_clusters=num_clusters, metric='precomputed', linkage='average'
        ).fit_predict(1 - init_sim)
        print(f"Initial clusters ({init_mode}): {client_cluster_assignments.tolist()}")

    local_models = [CIFAR_CNN().to(DEVICE) for _ in range(num_clients)]
    single_model_mb = calculate_model_size(local_models[0])
    print(f"Model size: {single_model_mb:.3f} MB ({sum(p.numel() for p in local_models[0].parameters()):,} params)")
    total_comm_mb = 0

    performance_history = []
    aggregation_times = []

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    for round_num in tqdm(range(NUM_ROUNDS), desc=run_label):

        # --- Get scheduled weights ---
        a_eff, b_eff, g_eff = get_scheduled_weights(round_num, NUM_ROUNDS, schedule)

        # --- Re-clustering (FCMA only) ---
        if mode == 'FCMA' and round_num >= 3 and round_num % RECLUSTERING_INTERVAL == 0:
            prev_assign = client_cluster_assignments.copy()
            updates = np.array([
                (get_flat_params(local_models[i]) - get_flat_params(cluster_models[prev_assign[i]])).cpu().numpy()
                for i in range(num_clients)
            ])
            active_mask = ~np.all(updates == 0, axis=1)
            if np.sum(active_mask) > LOW_RANK_DIM:
                pca = PCA(n_components=LOW_RANK_DIM, random_state=SEED)
                pca.fit(updates[active_mask])
                s_model = calculate_s_model(updates, pca.components_.T)
                s_data = calculate_s_data(client_metadata)
                combined = a_eff * s_model + b_eff * s_data
                if g_eff > 0:
                    combined += g_eff * calculate_s_context(client_metadata)

                new_assign = AgglomerativeClustering(
                    n_clusters=num_clusters, metric='precomputed', linkage='average'
                ).fit_predict(1 - combined)

                # Stability: only move clients whose similarity to their current cluster
                # is significantly lower than their similarity to the proposed new cluster
                n_moved = 0
                for cid in range(num_clients):
                    if new_assign[cid] != prev_assign[cid]:
                        # Check if move is justified: similarity to new cluster > old + margin
                        old_cl_members = [j for j in range(num_clients) if prev_assign[j] == prev_assign[cid] and j != cid]
                        new_cl_members = [j for j in range(num_clients) if new_assign[j] == new_assign[cid] and j != cid]
                        old_sim = np.mean([combined[cid, j] for j in old_cl_members]) if old_cl_members else 0
                        new_sim = np.mean([combined[cid, j] for j in new_cl_members]) if new_cl_members else 0
                        if new_sim > old_sim + 0.05:  # Only move if clearly better
                            client_cluster_assignments[cid] = new_assign[cid]
                            n_moved += 1
                        # else: stay in current cluster
                    # If same assignment, keep it

                tqdm.write(f"  Round {round_num+1}: Re-clustered (moved {n_moved}/{num_clients} clients)")

        # --- Local training ---
        current_local = []
        n_active = 0
        for cid in range(num_clients):
            if len(client_datasets[cid]) == 0:
                current_local.append(copy.deepcopy(local_models[cid]))
                continue
            n_active += 1

            if mode == 'FedAvg':
                m = copy.deepcopy(global_model)
            else:
                cidx = client_cluster_assignments[cid]
                if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
                    prev_cidx = prev_assign[cid]
                    if cidx != prev_cidx:
                        old_sd = local_models[cid].state_dict()
                        new_sd = cluster_models[cidx].state_dict()
                        blended = {k: MODEL_BLEND_WEIGHT * old_sd[k] + (1 - MODEL_BLEND_WEIGHT) * new_sd[k]
                                   for k in old_sd}
                        m = CIFAR_CNN().to(DEVICE)
                        m.load_state_dict(blended)
                    else:
                        m = copy.deepcopy(cluster_models[cidx])
                else:
                    m = copy.deepcopy(cluster_models[cidx])

            lr = max(LEARNING_RATE * (0.98 ** (round_num // 5)), MIN_LR)
            optimizer = optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            m.train()

            for _ in range(LOCAL_EPOCHS):
                for data, target in client_loaders[cid]:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = m(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()
            current_local.append(m)

        local_models = current_local
        total_comm_mb += 2 * n_active * single_model_mb

        # --- Aggregation ---
        t0 = time.time()
        if mode == 'FedAvg':
            active_models = [local_models[i] for i in range(num_clients) if len(client_datasets[i]) > 0]
            if active_models:
                global_model = federated_averaging(active_models)
        else:
            for cl_id in range(num_clusters):
                models_in = [local_models[i] for i in range(num_clients)
                             if client_cluster_assignments[i] == cl_id and len(client_datasets[i]) > 0]
                if models_in:
                    if agg_mode == 'fedma':
                        agg = intra_cluster_fedma_cnn(models_in, cluster_models[cl_id], SIMILARITY_THRESHOLD)
                    else:
                        agg = federated_averaging(models_in)
                    if agg:
                        cluster_models[cl_id] = agg
        aggregation_times.append(time.time() - t0)

        # --- Evaluate (FAIR: both methods on global test set) ---
        if mode == 'FedAvg':
            acc = evaluate(global_model, global_test_loader)
        else:
            # For FCMA: evaluate each cluster model on global test, weighted by cluster size
            cluster_accs = []
            cluster_sizes = []
            for cl_id in range(num_clusters):
                members = [i for i in range(num_clients) if client_cluster_assignments[i] == cl_id and len(client_datasets[i]) > 0]
                if members:
                    cl_acc = evaluate(cluster_models[cl_id], global_test_loader)
                    cluster_accs.append(cl_acc)
                    cluster_sizes.append(len(members))
            # Weighted average by cluster size
            if cluster_accs:
                total_members = sum(cluster_sizes)
                acc = sum(a * s for a, s in zip(cluster_accs, cluster_sizes)) / total_members
            else:
                acc = 0.0

        if (round_num + 1) % 10 == 0 or round_num == 0:
            tqdm.write(f"  Round {round_num+1:3d}: Acc={acc:.4f}  Comm={total_comm_mb:.1f}MB")

        performance_history.append({'round': round_num + 1, 'accuracy': acc})

        if torch.cuda.is_available() and round_num % 5 == 0:
            torch.cuda.empty_cache()

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print(f"\n--- Final: {run_label} ---")
    print(f"Avg aggregation: {np.mean(aggregation_times):.4f}s/round")
    print(f"Total comm: {total_comm_mb:.1f} MB")

    # Global test accuracy
    if mode == 'FedAvg':
        global_acc = evaluate(global_model, global_test_loader)
    else:
        # Best cluster model on global test
        best_acc = max(evaluate(m, global_test_loader) for m in cluster_models)
        global_acc = best_acc

    # Personalized accuracy
    pers_accs = []
    for cid in range(num_clients):
        if len(client_datasets[cid]) == 0:
            continue
        if mode == 'FedAvg':
            m_eval = global_model
        else:
            m_eval = cluster_models[client_cluster_assignments[cid]]
        a = evaluate_per_client(m_eval, client_datasets[cid])
        pers_accs.append(a)

    avg_pers = np.mean(pers_accs) if pers_accs else 0
    print(f"  Global Test Acc: {global_acc:.4f}")
    print(f"  Personalized Acc: {avg_pers:.4f}")

    # Rounds to 90%
    rounds_to_90 = -1
    for h in performance_history:
        if h['accuracy'] >= 0.90:
            rounds_to_90 = h['round']
            break
    print(f"  Rounds to 90%: {rounds_to_90 if rounds_to_90 > 0 else 'Not reached'}")

    # --- Plots ---
    hist_df = pd.DataFrame(performance_history)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_df['round'], hist_df['accuracy'], 'o-', markersize=2)
    plt.title(f'{run_label}')
    plt.xlabel('Round'); plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_folder, 'convergence.png'), dpi=150)
    plt.close()

    # Save summary
    with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
        f.write(f"=== {run_label} ===\n")
        f.write(f"Dataset: CIFAR-10\nClients: {num_clients}\nDirichlet alpha: {alpha}\n")
        f.write(f"Clusters: {num_clusters}\nMode: {mode}\n")
        f.write(f"Agg: {agg_mode} | Init: {init_mode} | Schedule: {schedule}\n")
        f.write(f"Global Acc: {global_acc:.4f}\nPers Acc: {avg_pers:.4f}\n")
        f.write(f"Rounds to 90%: {rounds_to_90}\nComm: {total_comm_mb:.1f} MB\n")

    print(f"Results saved to {results_folder}/")
    return {
        'run': run_label, 'global_acc': global_acc, 'pers_acc': avg_pers,
        'rounds_to_90': rounds_to_90, 'history': performance_history,
    }


# ==============================================================================
# --- MAIN ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Fed-CMA on CIFAR-10')
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet concentration')
    parser.add_argument('--mode', type=str, default='both', choices=['FedAvg', 'FCMA', 'both'])
    parser.add_argument('--agg_mode', type=str, default='fedma', choices=['fedma', 'fedavg'],
                        help='C1: intra-cluster aggregation method')
    parser.add_argument('--init_mode', type=str, default='context', choices=['context', 'random', 'model'],
                        help='C2: initial clustering method')
    parser.add_argument('--schedule', type=str, default='fixed', choices=['fixed', 'linear', 'cosine'],
                        help='C3: weight scheduling strategy')
    parser.add_argument('--rounds', type=int, default=100)
    args = parser.parse_args()

    global NUM_ROUNDS
    NUM_ROUNDS = args.rounds

    all_results = []
    modes = ['FedAvg', 'FCMA'] if args.mode == 'both' else [args.mode]

    for mode in modes:
        print(f"\n{'#'*60}")
        print(f"#  {mode} | {args.clients} clients | alpha={args.alpha}")
        print(f"{'#'*60}")

        agg = args.agg_mode if mode == 'FCMA' else 'fedavg'
        init = args.init_mode if mode == 'FCMA' else 'random'
        sched = args.schedule if mode == 'FCMA' else 'fixed'

        result = run_experiment(mode, args.clients, args.alpha, agg, init, sched)
        all_results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY (CIFAR-10)")
    print(f"{'='*60}")
    print(f"{'Run':<55s} | {'Acc':>5s} | {'Pers':>5s} | {'R90':>4s}")
    print("-" * 78)
    for r in all_results:
        r90 = str(r['rounds_to_90']) if r['rounds_to_90'] > 0 else 'N/A'
        print(f"{r['run']:<55s} | {r['global_acc']:>5.3f} | {r['pers_acc']:>5.3f} | {r90:>4s}")

    # Comparison plot
    if len(all_results) > 1:
        plt.figure(figsize=(12, 6))
        for r in all_results:
            h = pd.DataFrame(r['history'])
            plt.plot(h['round'], h['accuracy'], label=r['run'], linewidth=1.5)
        plt.title(f'CIFAR-10: FedAvg vs Fed-CMA (N={args.clients}, alpha={args.alpha})')
        plt.xlabel('Round'); plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3); plt.legend(fontsize=7)
        plt.savefig('cifar10_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nComparison plot saved to cifar10_comparison.png")


if __name__ == '__main__':
    main()
