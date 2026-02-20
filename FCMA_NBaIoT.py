# ==============================================================================
# FCMA_NBaIoT.py
# Fed-CMA experiment on N-BaIoT IoT Botnet Detection dataset
#
# Key differences from the Car Hacking (CAN bus) version:
#   - MODEL: MLP instead of LSTM (features are pre-computed statistics, not
#     raw time-series). The 115 features are already temporal aggregates computed
#     over 5 time windows by the N-BaIoT feature extractor.
#   - PARTITIONING: By device_id (9 real IoT devices = 9 natural FL clients).
#     Each device has genuinely different traffic patterns → real feature skew.
#     Can also scale to 18+ clients by splitting large devices temporally.
#   - CONTEXT: Derived from real device metadata:
#       S_device_type: indicator (camera vs doorbell vs thermostat vs webcam)
#       S_botnet: Jaccard similarity of botnet family exposure (Mirai/Bashlite)
#       S_attack: Jaccard similarity of attack subtypes (ack, syn, udp, combo...)
#   - MATCHING: Standard FC matched averaging (not LSTM-specific), which is the
#     simpler case from the FedMA paper.
#
# Usage:
#   python FCMA_NBaIoT.py --data nbaiot_processed.csv
# ==============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

print("Libraries imported successfully.")

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
NUM_ROUNDS = 100
LOCAL_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.001
SEED = 42

GRADIENT_CLIP_NORM = 1.0
MIN_LR = 0.00005

RECLUSTERING_INTERVAL = 20
LOW_RANK_DIM = 10
SIMILARITY_THRESHOLD = 0.1

# Similarity weights (alpha + beta + gamma = 1.0)
ALPHA = 0.25    # Model update similarity
BETA = 0.45     # Data distribution similarity
GAMMA = 0.30    # Context similarity (higher than CAN — device type is very informative)
MODEL_BLEND_WEIGHT = 0.5

# Context sub-weights
W_DEVICE_TYPE = 0.35
W_BOTNET = 0.30
W_ATTACK = 0.35

POISONING_RATE = 0.0
POISONING_FLIP_RATE = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def get_num_clusters(num_clients):
    return max(2, min(10, int(np.sqrt(num_clients))))


# ==============================================================================
# --- DATA LOADING ---
# ==============================================================================
def load_nbaiot_data(data_path):
    """Load the preprocessed N-BaIoT CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found. Run preprocess_nbaiot.py first.")
    print(f"Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Devices: {df['device_id'].nunique()}")
    print(f"  Attack rate: {df['BinaryLabel'].mean()*100:.1f}%")
    return df


# ==============================================================================
# --- MODEL (MLP for tabular features) ---
# ==============================================================================
class IoT_MLP(nn.Module):
    """
    3-layer MLP for N-BaIoT intrusion detection.
    Input: 115 pre-computed statistical features.
    Architecture chosen to be comparable in parameter count to the LSTM used
    for CAN bus (~37K params), enabling fair communication cost comparison.
    """
    def __init__(self, input_dim):
        super(IoT_MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        return self.fc_out(x)


# ==============================================================================
# --- DATASET ---
# ==============================================================================
class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].unsqueeze(-1)


# ==============================================================================
# --- PARTITIONING ---
# ==============================================================================
def _get_device_category(device_name):
    """Map a device name to its category."""
    device_type_map = {
        'doorbell': 'Doorbell', 'thermostat': 'Thermostat',
        'baby_monitor': 'Monitor', 'baby monitor': 'Monitor',
        'security_camera': 'Camera', 'security camera': 'Camera',
        'webcam': 'Camera',
    }
    name_lower = device_name.lower()
    for key, val in device_type_map.items():
        if key in name_lower:
            return val
    return 'Other'


def _build_client_metadata(cdf, cid):
    """Extract metadata from a client's data chunk."""
    if len(cdf) == 0:
        return {
            'client_id': cid, 'device_name': 'Empty', 'device_category': 'Other',
            'attack_exposure': set(), 'botnet_families': set(), 'is_poisoned': False,
        }
    attack_types = set(cdf.loc[cdf['BinaryLabel'] == 1, 'Label'].unique())
    botnet_families = set(cdf.loc[cdf['attack_family'] != 'none', 'attack_family'].unique())
    primary_device = cdf['device_name'].value_counts().index[0]
    return {
        'client_id': cid,
        'device_name': primary_device,
        'device_category': _get_device_category(primary_device),
        'attack_exposure': attack_types,
        'botnet_families': botnet_families,
        'is_poisoned': False,
    }


def _build_loaders(client_dfs, feature_cols, num_features):
    """Convert list of DataFrames into train/test loaders + metadata."""
    train_loaders, test_loaders = [], []
    client_metadata_list = []
    num_clients = len(client_dfs)

    for cid in range(num_clients):
        cdf = client_dfs[cid]
        client_metadata_list.append(_build_client_metadata(cdf, cid))

        if len(cdf) < 100:
            train_loaders.append(DataLoader(TabularDataset(
                np.zeros((0, num_features)), np.zeros(0)), batch_size=BATCH_SIZE))
            test_loaders.append(DataLoader(TabularDataset(
                np.zeros((0, num_features)), np.zeros(0)), batch_size=BATCH_SIZE))
            continue

        try:
            tr, te = train_test_split(cdf, test_size=0.2, random_state=SEED, stratify=cdf['BinaryLabel'])
        except ValueError:
            tr, te = train_test_split(cdf, test_size=0.2, random_state=SEED)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[feature_cols].values)
        X_te = scaler.transform(te[feature_cols].values)
        train_loaders.append(DataLoader(
            TabularDataset(X_tr, tr['BinaryLabel'].values), batch_size=BATCH_SIZE, shuffle=True))
        test_loaders.append(DataLoader(
            TabularDataset(X_te, te['BinaryLabel'].values), batch_size=BATCH_SIZE, shuffle=False))

    return train_loaders, test_loaders, client_metadata_list


def partition_by_device(df, num_clients):
    """
    STRATEGY 1: Natural device-based partition.
    9 devices = 9 natural clients. If num_clients > 9, split largest devices
    temporally (simulating multiple instances of the same device type).
    Best for: "realistic deployment" baseline in thesis.
    """
    meta_cols = ['device_id', 'device_name', 'Label', 'BinaryLabel', 'attack_family']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    num_features = len(feature_cols)
    device_ids = sorted(df['device_id'].unique())
    num_devices = len(device_ids)

    print(f"Partitioning (device-natural): {num_devices} devices → {num_clients} clients")

    if num_clients <= num_devices:
        client_dfs = []
        for cid in range(num_clients):
            assigned = [d for i, d in enumerate(device_ids) if i % num_clients == cid]
            client_dfs.append(df[df['device_id'].isin(assigned)])
    else:
        client_dfs = [df[df['device_id'] == d].copy() for d in device_ids]
        remaining = num_clients - num_devices
        while remaining > 0:
            sizes = [len(d) for d in client_dfs]
            largest_idx = int(np.argmax(sizes))
            largest_df = client_dfs[largest_idx]
            mid = len(largest_df) // 2
            client_dfs[largest_idx] = largest_df.iloc[:mid]
            client_dfs.append(largest_df.iloc[mid:])
            remaining -= 1
        client_dfs = client_dfs[:num_clients]

    train_loaders, test_loaders, metadata = _build_loaders(client_dfs, feature_cols, num_features)
    _print_partition_summary(metadata, train_loaders, num_clients)
    return train_loaders, test_loaders, metadata, num_features


def partition_dirichlet(df, num_clients, alpha=0.5):
    """
    STRATEGY 2: Dirichlet-based non-IID partition (the FL standard).

    Samples class proportions from Dir(alpha) independently for each client.
    - alpha=0.1: extreme heterogeneity (some clients get almost all-benign,
      others almost all-attack, with very skewed attack subtypes)
    - alpha=0.5: moderate heterogeneity (the standard in FedAvg/FedMA papers)
    - alpha=1.0: mild heterogeneity
    - alpha=100: near-IID

    Importantly, this preserves device information in the metadata — each
    sample retains its device_name, so S_context can still use device type
    similarity based on the dominant device in each client's partition.

    Reference: FedMA (Wang et al., 2020) uses Dir(0.5) for CIFAR-10.
    """
    meta_cols = ['device_id', 'device_name', 'Label', 'BinaryLabel', 'attack_family']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    num_features = len(feature_cols)

    print(f"Partitioning (Dirichlet α={alpha}): {len(df):,} samples → {num_clients} clients")

    rng = np.random.default_rng(SEED)

    # Use the full attack subtype labels for partitioning (not just binary)
    # This creates both label skew AND attack-type skew
    unique_labels = sorted(df['Label'].unique())
    num_classes = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    df_labels = df['Label'].map(label_to_idx).values

    # Sample proportions from Dirichlet for each class
    # proportions[k] is a vector of length num_clients summing to 1
    client_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        class_indices = np.where(df_labels == k)[0]
        rng.shuffle(class_indices)

        # Sample proportions from Dir(alpha)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to actual sample counts
        proportions = (proportions * len(class_indices)).astype(int)
        # Handle rounding: add remainder to random clients
        remainder = len(class_indices) - proportions.sum()
        if remainder > 0:
            bonus_clients = rng.choice(num_clients, size=remainder, replace=False)
            proportions[bonus_clients] += 1
        elif remainder < 0:
            # Remove excess from largest allocations
            for _ in range(-remainder):
                largest = np.argmax(proportions)
                proportions[largest] -= 1

        # Distribute indices
        offset = 0
        for cid in range(num_clients):
            n_samples = proportions[cid]
            client_indices[cid].extend(class_indices[offset:offset + n_samples].tolist())
            offset += n_samples

    # Build client DataFrames
    client_dfs = [df.iloc[indices].copy() if indices else pd.DataFrame(columns=df.columns)
                  for indices in client_indices]

    train_loaders, test_loaders, metadata = _build_loaders(client_dfs, feature_cols, num_features)

    # Log heterogeneity statistics
    print(f"\n  Dirichlet partition stats (α={alpha}):")
    sizes = [len(indices) for indices in client_indices]
    print(f"  Samples per client: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.0f}, std={np.std(sizes):.0f}")
    attack_rates = []
    for cid in range(num_clients):
        if len(client_dfs[cid]) > 0:
            rate = client_dfs[cid]['BinaryLabel'].mean()
            attack_rates.append(rate)
    if attack_rates:
        print(f"  Attack rate per client: min={min(attack_rates):.1%}, "
              f"max={max(attack_rates):.1%}, std={np.std(attack_rates):.3f}")

    _print_partition_summary(metadata, train_loaders, num_clients)
    return train_loaders, test_loaders, metadata, num_features


def partition_by_attack_type(df, num_clients):
    """
    STRATEGY 3: Attack-type partition.
    Each client specialises in a specific attack subtype + benign traffic.
    Creates extreme label skew — ideal for demonstrating that Fed-CMA's
    clustering correctly groups clients with similar threat profiles.

    If num_clients > num_attack_types: multiple clients share an attack type
    but get different subsets of the data (temporal split).
    """
    meta_cols = ['device_id', 'device_name', 'Label', 'BinaryLabel', 'attack_family']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    num_features = len(feature_cols)

    attack_labels = sorted(df.loc[df['BinaryLabel'] == 1, 'Label'].unique())
    benign_df = df[df['BinaryLabel'] == 0]
    num_attack_types = len(attack_labels)

    print(f"Partitioning (attack-type): {num_attack_types} attack types → {num_clients} clients")

    # Distribute benign data evenly across all clients
    benign_per_client = len(benign_df) // num_clients
    rng = np.random.default_rng(SEED)
    benign_shuffled = benign_df.sample(frac=1, random_state=SEED)

    client_dfs = []
    for cid in range(num_clients):
        # Each client gets a slice of benign
        b_start = cid * benign_per_client
        b_end = b_start + benign_per_client if cid < num_clients - 1 else len(benign_shuffled)
        client_benign = benign_shuffled.iloc[b_start:b_end]

        # Assign attack types round-robin
        attack_type = attack_labels[cid % num_attack_types]
        attack_data = df[df['Label'] == attack_type]

        # If multiple clients share an attack type, split the attack data
        clients_with_this_type = [c for c in range(num_clients)
                                  if attack_labels[c % num_attack_types] == attack_type]
        split_idx = clients_with_this_type.index(cid)
        n_splits = len(clients_with_this_type)
        chunk_size = len(attack_data) // n_splits
        a_start = split_idx * chunk_size
        a_end = a_start + chunk_size if split_idx < n_splits - 1 else len(attack_data)
        client_attack = attack_data.iloc[a_start:a_end]

        client_dfs.append(pd.concat([client_benign, client_attack], ignore_index=True))

    train_loaders, test_loaders, metadata = _build_loaders(client_dfs, feature_cols, num_features)
    _print_partition_summary(metadata, train_loaders, num_clients)
    return train_loaders, test_loaders, metadata, num_features


def _print_partition_summary(metadata, train_loaders, num_clients):
    """Print a summary of the partition."""
    print(f"\n--- Client Metadata ---")
    for m in metadata:
        atk = ', '.join(sorted(m['attack_exposure']))[:50] if m['attack_exposure'] else 'None'
        bots = ', '.join(sorted(m['botnet_families'])) if m['botnet_families'] else 'None'
        print(f"  Client {m['client_id']:2d}: {m['device_name'][:25]:<25s} "
              f"Cat={m['device_category']:<10s} Botnets={bots:<15s} Attacks={atk}")
    active = sum(1 for l in train_loaders if len(l.dataset) > 0)
    print(f"\n  {active}/{num_clients} clients with data")


# ==============================================================================
# --- CONTEXT SIMILARITY ---
# ==============================================================================
def calculate_s_context(client_metadata,
                        w_device=W_DEVICE_TYPE, w_botnet=W_BOTNET, w_attack=W_ATTACK):
    """
    Contextual similarity for IoT devices.
    S_context = w_device * S_device_type + w_botnet * S_botnet + w_attack * S_attack

    S_device_type: indicator (same device category)
    S_botnet: Jaccard over botnet families {mirai, gafgyt}
    S_attack: Jaccard over attack subtypes (mirai_ack, gafgyt_combo, etc.)
    """
    n = len(client_metadata)
    s = np.zeros((n, n))
    for i in range(n):
        s[i, i] = 1.0
        for j in range(i + 1, n):
            mi, mj = client_metadata[i], client_metadata[j]

            # Device type similarity
            dev_sim = 1.0 if mi['device_category'] == mj['device_category'] else 0.0

            # Botnet family similarity (Jaccard)
            bi, bj = mi['botnet_families'], mj['botnet_families']
            if not bi and not bj:
                bot_sim = 1.0
            elif not bi or not bj:
                bot_sim = 0.0
            else:
                bot_sim = len(bi & bj) / len(bi | bj)

            # Attack subtype similarity (Jaccard)
            ai, aj = mi['attack_exposure'], mj['attack_exposure']
            if not ai and not aj:
                att_sim = 1.0
            elif not ai or not aj:
                att_sim = 0.0
            else:
                att_sim = len(ai & aj) / len(ai | aj)

            sim = w_device * dev_sim + w_botnet * bot_sim + w_attack * att_sim
            s[i, j] = s[j, i] = sim
    return s


# ==============================================================================
# --- POISONING ---
# ==============================================================================
def apply_label_flipping(client_dataloaders, client_metadata, poisoning_rate,
                         flip_rate=1.0, seed=42):
    rng = np.random.default_rng(seed)
    num_to_poison = max(0, int(len(client_dataloaders) * poisoning_rate))
    if num_to_poison == 0:
        return set()
    eligible = [i for i, l in enumerate(client_dataloaders) if len(l.dataset) > 0]
    poisoned_ids = set(rng.choice(eligible, size=min(num_to_poison, len(eligible)), replace=False))
    for i in poisoned_ids:
        client_metadata[i]['is_poisoned'] = True
        labels = client_dataloaders[i].dataset.labels
        n_flip = int(len(labels) * flip_rate)
        flip_idx = rng.choice(len(labels), size=n_flip, replace=False)
        labels[flip_idx] = 1.0 - labels[flip_idx]
    return poisoned_ids


def calculate_poisoning_isolation_rate(assignments, poisoned_ids, num_clients):
    if not poisoned_ids:
        return {'isolation_rate': 1.0, 'contamination_rate': 0.0}
    clean_ids = set(range(num_clients)) - poisoned_ids
    isolated = sum(1 for p in poisoned_ids if p < len(assignments) and
                   not any(assignments[c] == assignments[p] for c in clean_ids if c < len(assignments)))
    poisoned_clusters = set(assignments[p] for p in poisoned_ids if p < len(assignments))
    contaminated = sum(1 for c in clean_ids if c < len(assignments) and assignments[c] in poisoned_clusters)
    return {
        'isolation_rate': isolated / len(poisoned_ids),
        'contamination_rate': contaminated / len(clean_ids) if clean_ids else 0,
    }


# ==============================================================================
# --- FL HELPERS ---
# ==============================================================================
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def calculate_model_size(model):
    p = sum(param.nelement() * param.element_size() for param in model.parameters())
    b = sum(buf.nelement() * buf.element_size() for buf in model.buffers())
    return (p + b) / 1024**2


def calculate_s_data(client_dataloaders):
    hists = []
    for loader in client_dataloaders:
        if len(loader.dataset) == 0:
            hists.append(np.array([0.5, 0.5]))
            continue
        labels = loader.dataset.labels.numpy()
        total = len(labels)
        hists.append(np.array([np.sum(labels == 0) / total, np.sum(labels == 1) / total]))
    n = len(hists)
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = 1 - jensenshannon(hists[i], hists[j])
            s[i, j] = s[j, i] = sim
    return s


def calculate_s_model(model_updates, M):
    proj = model_updates @ M
    norm = np.linalg.norm(proj, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    cosine_sim = (proj @ proj.T) / (norm @ norm.T)
    return np.clip(cosine_sim, 0, 1)


def evaluate(model, test_loader, return_metrics=False, return_preds=False):
    model.eval()
    all_preds, all_targets = [], []
    if len(test_loader.dataset) == 0:
        if return_preds:
            return np.array([]), np.array([])
        return (0.0, 0.0, 0.0, 0.0) if return_metrics else 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    if not all_targets:
        if return_preds:
            return np.array([]), np.array([])
        return (0.0, 0.0, 0.0, 0.0) if return_metrics else 0.0
    if return_preds:
        return np.array(all_targets), np.array(all_preds)
    accuracy = accuracy_score(all_targets, all_preds)
    if return_metrics:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0)
        return accuracy, precision, recall, f1
    return accuracy


# ==============================================================================
# --- AGGREGATION ---
# ==============================================================================
def federated_averaging(models, input_dim):
    if not models:
        return None
    avg_sd = copy.deepcopy(models[0].state_dict())
    for key in avg_sd:
        avg_sd[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(0)
    agg = IoT_MLP(input_dim).to(DEVICE)
    agg.load_state_dict(avg_sd)
    return agg


def intra_cluster_fedma_mlp(cluster_models, ref_model, threshold):
    """
    FedMA for deep FC networks (FedMA paper Section 2.2, Equation 4).
    Recursively matches neurons layer by layer, propagating permutations.

    For an MLP with layers fc1→fc2→fc3→fc_out:
    1. Match fc1 neurons (output dim) using their weight signatures
    2. Propagate fc1 permutation to fc2 input dim, then match fc2 neurons
    3. Propagate fc2 permutation to fc3 input dim, then match fc3 neurons
    4. fc_out: just average (no permutation, it's the output layer)

    BatchNorm params are permuted to match their corresponding FC layer.
    """
    if not cluster_models:
        return None
    if len(cluster_models) == 1:
        return copy.deepcopy(cluster_models[0])

    ref_sd = ref_model.state_dict()
    n_models = len(cluster_models)
    all_sds = [m.state_dict() for m in cluster_models]
    agg_sd = {}
    input_dim = ref_model.input_dim

    # Layer structure: fc1(in→128), fc2(128→64), fc3(64→32), fc_out(32→1)
    layers = [
        {'fc': 'fc1', 'bn': 'bn1', 'out_dim': 128, 'in_dim': input_dim},
        {'fc': 'fc2', 'bn': 'bn2', 'out_dim': 64,  'in_dim': 128},
        {'fc': 'fc3', 'bn': 'bn3', 'out_dim': 32,  'in_dim': 64},
    ]

    prev_perms = [None] * n_models  # No input permutation for first layer

    for layer_info in layers:
        fc_name = layer_info['fc']
        bn_name = layer_info['bn']
        out_dim = layer_info['out_dim']

        # Build matching signatures: for each client, take fc weight
        # (with input dim permuted from previous layer) and find best matching
        ref_w = ref_sd[f'{fc_name}.weight']  # (out_dim, in_dim)

        current_perms = []
        for m_idx in range(n_models):
            client_w = all_sds[m_idx][f'{fc_name}.weight'].clone()

            # Apply previous layer's permutation to input dimension
            if prev_perms[m_idx] is not None:
                inv_prev = np.argsort(prev_perms[m_idx])
                client_w = client_w[:, inv_prev]

            # Match neurons (rows of weight matrix)
            cost = 1 - torch.nn.functional.cosine_similarity(
                ref_w.unsqueeze(1), client_w.unsqueeze(0), dim=2
            )
            _, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            current_perms.append(col_ind)

        # Aggregate with permutations applied
        sum_w = torch.zeros_like(ref_sd[f'{fc_name}.weight']).float()
        sum_b = torch.zeros_like(ref_sd[f'{fc_name}.bias']).float()
        # BatchNorm: weight, bias, running_mean, running_var
        sum_bn_w = torch.zeros_like(ref_sd[f'{bn_name}.weight']).float()
        sum_bn_b = torch.zeros_like(ref_sd[f'{bn_name}.bias']).float()
        sum_bn_rm = torch.zeros_like(ref_sd[f'{bn_name}.running_mean']).float()
        sum_bn_rv = torch.zeros_like(ref_sd[f'{bn_name}.running_var']).float()

        for m_idx in range(n_models):
            inv_perm = np.argsort(current_perms[m_idx])
            sd = all_sds[m_idx]

            w = sd[f'{fc_name}.weight'].clone()
            # Permute input dim from previous layer
            if prev_perms[m_idx] is not None:
                inv_prev = np.argsort(prev_perms[m_idx])
                w = w[:, inv_prev]
            # Permute output dim (current layer's neurons)
            w = w[inv_perm, :]
            sum_w += w

            # Bias: permute with current layer
            sum_b += sd[f'{fc_name}.bias'][inv_perm]

            # BatchNorm: aligned with current layer's neurons
            sum_bn_w += sd[f'{bn_name}.weight'][inv_perm]
            sum_bn_b += sd[f'{bn_name}.bias'][inv_perm]
            sum_bn_rm += sd[f'{bn_name}.running_mean'][inv_perm]
            sum_bn_rv += sd[f'{bn_name}.running_var'][inv_perm]

        agg_sd[f'{fc_name}.weight'] = sum_w / n_models
        agg_sd[f'{fc_name}.bias'] = sum_b / n_models
        agg_sd[f'{bn_name}.weight'] = sum_bn_w / n_models
        agg_sd[f'{bn_name}.bias'] = sum_bn_b / n_models
        agg_sd[f'{bn_name}.running_mean'] = sum_bn_rm / n_models
        agg_sd[f'{bn_name}.running_var'] = sum_bn_rv / n_models
        agg_sd[f'{bn_name}.num_batches_tracked'] = ref_sd[f'{bn_name}.num_batches_tracked']

        prev_perms = current_perms

    # Output layer: permute input dim from last hidden layer, average output
    sum_w = torch.zeros_like(ref_sd['fc_out.weight']).float()
    sum_b = torch.zeros_like(ref_sd['fc_out.bias']).float()
    for m_idx in range(n_models):
        inv_prev = np.argsort(prev_perms[m_idx])
        sum_w += all_sds[m_idx]['fc_out.weight'][:, inv_prev]
        sum_b += all_sds[m_idx]['fc_out.bias']
    agg_sd['fc_out.weight'] = sum_w / n_models
    agg_sd['fc_out.bias'] = sum_b / n_models

    agg = IoT_MLP(input_dim).to(DEVICE)
    agg.load_state_dict(agg_sd)
    return agg


# ==============================================================================
# --- MAIN EXPERIMENT ---
# ==============================================================================
def run_experiment(federated_mode, df, num_clients, partition_strategy='device',
                   dirichlet_alpha=0.5, use_context=True, poisoning_rate=0.0):
    num_clusters = get_num_clusters(num_clients)

    if federated_mode == 'FCMA' and use_context:
        a_eff, b_eff, g_eff = ALPHA, BETA, GAMMA
    elif federated_mode == 'FCMA' and not use_context:
        a_eff = ALPHA / (ALPHA + BETA)
        b_eff = BETA / (ALPHA + BETA)
        g_eff = 0.0
    else:
        a_eff, b_eff, g_eff = ALPHA, BETA, 0.0

    strat_label = {'device': 'dev', 'dirichlet': f'dir{dirichlet_alpha}',
                   'attack_type': 'atk'}[partition_strategy]
    ctx_label = "_ctx" if (federated_mode == 'FCMA' and use_context) else ""
    poison_label = f"_p{int(poisoning_rate*100)}" if poisoning_rate > 0 else ""
    run_label = f"{federated_mode}_N{num_clients}_{strat_label}{ctx_label}{poison_label}_NBaIoT"

    print(f"\n{'='*60}")
    print(f"  {run_label}  |  Clusters: {num_clusters}")
    if federated_mode == 'FCMA':
        print(f"  Weights: α={a_eff:.3f}  β={b_eff:.3f}  γ={g_eff:.3f}")
    print(f"  Partition: {partition_strategy}" +
          (f" (α={dirichlet_alpha})" if partition_strategy == 'dirichlet' else ''))
    print(f"{'='*60}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results_{run_label}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)

    # --- Partition & metadata ---
    if partition_strategy == 'device':
        train_loaders, test_loaders, client_metadata, num_features = partition_by_device(df, num_clients)
    elif partition_strategy == 'dirichlet':
        train_loaders, test_loaders, client_metadata, num_features = partition_dirichlet(df, num_clients, alpha=dirichlet_alpha)
    elif partition_strategy == 'attack_type':
        train_loaders, test_loaders, client_metadata, num_features = partition_by_attack_type(df, num_clients)
    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")
    input_dim = num_features

    poisoned_ids = apply_label_flipping(
        train_loaders, client_metadata, poisoning_rate, POISONING_FLIP_RATE, SEED)

    # Global test set (20% of all data)
    meta_cols = ['device_id', 'device_name', 'Label', 'BinaryLabel', 'attack_family']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    _, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['BinaryLabel'])
    X_test = StandardScaler().fit_transform(test_df[feature_cols].values)
    global_test_loader = DataLoader(
        TabularDataset(X_test, test_df['BinaryLabel'].values), batch_size=BATCH_SIZE)

    # Class weight
    label_counts = df['BinaryLabel'].value_counts()
    pos_weight = label_counts.get(0, 1) / label_counts.get(1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))

    # Print distribution
    print(f"\n--- Client Data Distribution ---")
    for cid in range(num_clients):
        loader = train_loaders[cid]
        if len(loader.dataset) == 0:
            print(f"  Client {cid:2d}: No data")
            continue
        labels = loader.dataset.labels.numpy()
        att_pct = labels.mean() * 100
        m = client_metadata[cid]
        print(f"  Client {cid:2d}: {len(labels):7d} samples | Attack: {att_pct:5.1f}% | "
              f"{m['device_category']:<8s} | {m['device_name'][:25]}")

    # --- Model init ---
    if federated_mode == 'FedAvg':
        global_model = IoT_MLP(input_dim).to(DEVICE)
        client_cluster_assignments = np.zeros(num_clients, dtype=int)
    else:
        cluster_models = [IoT_MLP(input_dim).to(DEVICE) for _ in range(num_clusters)]
        data_sim = calculate_s_data(train_loaders)
        if g_eff > 0:
            ctx_sim = calculate_s_context(client_metadata)
            init_sim = (b_eff / (b_eff + g_eff)) * data_sim + (g_eff / (b_eff + g_eff)) * ctx_sim
        else:
            init_sim = data_sim
        client_cluster_assignments = AgglomerativeClustering(
            n_clusters=num_clusters, metric='precomputed', linkage='average'
        ).fit_predict(1 - init_sim)
        print(f"Initial clusters: {client_cluster_assignments.tolist()}")

    local_models = [IoT_MLP(input_dim).to(DEVICE) for _ in range(num_clients)]
    single_model_mb = calculate_model_size(local_models[0])
    print(f"Model size: {single_model_mb:.3f} MB  ({sum(p.numel() for p in local_models[0].parameters()):,} params)")
    total_comm_mb = 0

    performance_history = []
    aggregation_times = []
    isolation_history = []

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    for round_num in tqdm(range(NUM_ROUNDS), desc=run_label):

        # --- Re-clustering ---
        if federated_mode == 'FCMA' and round_num >= 3 and round_num % RECLUSTERING_INTERVAL == 0:
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
                s_data = calculate_s_data(train_loaders)
                combined = a_eff * s_model + b_eff * s_data
                if g_eff > 0:
                    combined += g_eff * calculate_s_context(client_metadata)
                client_cluster_assignments = AgglomerativeClustering(
                    n_clusters=num_clusters, metric='precomputed', linkage='average'
                ).fit_predict(1 - combined)
                tqdm.write(f"  Round {round_num+1}: Re-clustered → {client_cluster_assignments.tolist()}")

                if poisoned_ids:
                    iso = calculate_poisoning_isolation_rate(
                        client_cluster_assignments, poisoned_ids, num_clients)
                    isolation_history.append({'round': round_num + 1, **iso})

        # --- Local training ---
        current_local = []
        n_active = 0
        for cid in range(num_clients):
            loader = train_loaders[cid]
            if len(loader.dataset) == 0:
                current_local.append(copy.deepcopy(local_models[cid]))
                continue
            n_active += 1

            if federated_mode == 'FedAvg':
                m = copy.deepcopy(global_model)
            else:
                cidx = client_cluster_assignments[cid]
                if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
                    prev_cidx = prev_assign[cid]
                    if cidx != prev_cidx:
                        old_sd = local_models[cid].state_dict()
                        new_sd = cluster_models[cidx].state_dict()
                        blended = {k: MODEL_BLEND_WEIGHT * old_sd[k] + (1 - MODEL_BLEND_WEIGHT) * new_sd[k] for k in old_sd}
                        m = IoT_MLP(input_dim).to(DEVICE)
                        m.load_state_dict(blended)
                    else:
                        m = copy.deepcopy(cluster_models[cidx])
                else:
                    m = copy.deepcopy(cluster_models[cidx])

            lr = max(LEARNING_RATE * (0.95 ** (round_num // 10)), MIN_LR)
            optimizer = optim.Adam(m.parameters(), lr=lr)
            m.train()

            for _ in range(LOCAL_EPOCHS):
                for data, target in loader:
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
        if federated_mode == 'FedAvg':
            active_models = [local_models[i] for i in range(num_clients) if len(train_loaders[i].dataset) > 0]
            if active_models:
                global_model = federated_averaging(active_models, input_dim)
        else:
            for cl_id in range(num_clusters):
                models_in = [local_models[i] for i in range(num_clients)
                             if client_cluster_assignments[i] == cl_id and len(train_loaders[i].dataset) > 0]
                if models_in:
                    agg = intra_cluster_fedma_mlp(models_in, cluster_models[cl_id], SIMILARITY_THRESHOLD)
                    if agg:
                        cluster_models[cl_id] = agg
        aggregation_times.append(time.time() - t0)

        # --- Evaluate (personalized for FCMA) ---
        if federated_mode == 'FedAvg':
            acc, pre, rec, f1 = evaluate(global_model, global_test_loader, return_metrics=True)
        else:
            round_accs, round_f1s = [], []
            for cid in range(num_clients):
                if len(test_loaders[cid].dataset) == 0:
                    continue
                cl = client_cluster_assignments[cid]
                a, p, r, f = evaluate(cluster_models[cl], test_loaders[cid], return_metrics=True)
                round_accs.append(a)
                round_f1s.append(f)
            acc = np.mean(round_accs) if round_accs else 0
            f1 = np.mean(round_f1s) if round_f1s else 0

        if (round_num + 1) % 10 == 0 or round_num == 0:
            tqdm.write(f"  Round {round_num+1:3d}: Acc={acc:.4f}  F1={f1:.4f}  Comm={total_comm_mb:.1f}MB")

        performance_history.append({'round': round_num + 1, 'accuracy': acc, 'f1': f1})

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print(f"\n--- Final: {run_label} ---")
    print(f"Avg aggregation: {np.mean(aggregation_times):.4f}s/round")
    print(f"Total comm: {total_comm_mb:.1f} MB")

    local_accs, local_f1s = [], []
    for cid in range(num_clients):
        if len(test_loaders[cid].dataset) == 0:
            continue
        if federated_mode == 'FedAvg':
            m_eval = global_model
        else:
            m_eval = cluster_models[client_cluster_assignments[cid]]
        a, p, r, f = evaluate(m_eval, test_loaders[cid], return_metrics=True)
        local_accs.append(a); local_f1s.append(f)

    avg_acc = np.mean(local_accs) if local_accs else 0
    avg_f1 = np.mean(local_f1s) if local_f1s else 0
    print(f"  Personalized Acc: {avg_acc:.4f}  F1: {avg_f1:.4f}")

    # Plots
    hist_df = pd.DataFrame(performance_history)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_df['round'], hist_df['accuracy'], 'o-', label='Accuracy', markersize=2)
    plt.plot(hist_df['round'], hist_df['f1'], 'x--', label='F1-Score', markersize=2)
    plt.title(f'{run_label}'); plt.xlabel('Round'); plt.ylabel('Performance')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(results_folder, 'convergence.png'), dpi=150); plt.close()

    if federated_mode == 'FCMA' and g_eff > 0:
        ctx_sim = calculate_s_context(client_metadata)
        plt.figure(figsize=(8, 6))
        labels_list = [m['device_category'][:6] for m in client_metadata]
        sns.heatmap(ctx_sim, cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=labels_list, yticklabels=labels_list)
        plt.title(f'S_context: {run_label}')
        plt.savefig(os.path.join(results_folder, 'context_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()

    with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
        f.write(f"=== {run_label} ===\n")
        f.write(f"Dataset: N-BaIoT\nClients: {num_clients}\nClusters: {num_clusters}\n")
        f.write(f"Rounds: {NUM_ROUNDS}\nMode: {federated_mode}\n")
        f.write(f"Pers Acc: {avg_acc:.4f}\nPers F1: {avg_f1:.4f}\nComm: {total_comm_mb:.1f} MB\n")

    print(f"Results saved to {results_folder}/")
    return {'run': run_label, 'pers_accuracy': avg_acc, 'pers_f1': avg_f1, 'history': performance_history}


# ==============================================================================
# --- MAIN ---
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Fed-CMA on N-BaIoT')
    parser.add_argument('--data', type=str, default='nbaiot_processed.csv')
    parser.add_argument('--clients', type=int, nargs='+', default=[9, 20])
    parser.add_argument('--partition', type=str, default='device',
                        choices=['device', 'dirichlet', 'attack_type'],
                        help='Partition strategy: device (natural), dirichlet (standard FL), attack_type (extreme skew)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter (only used with --partition dirichlet)')
    args = parser.parse_args()

    df = load_nbaiot_data(args.data)
    all_results = []

    for n_clients in args.clients:
        for mode in ['FedAvg', 'FCMA']:
            print(f"\n{'#'*60}")
            print(f"#  {mode} | {n_clients} clients | {args.partition} partition")
            print(f"{'#'*60}")
            result = run_experiment(
                mode, df, n_clients,
                partition_strategy=args.partition,
                dirichlet_alpha=args.alpha,
                use_context=True, poisoning_rate=0.0)
            all_results.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY (N-BaIoT)")
    print(f"{'='*60}")
    print(f"{'Run':<50s} | {'Acc':>6s} | {'F1':>6s}")
    print("-" * 68)
    for r in all_results:
        print(f"{r['run']:<50s} | {r['pers_accuracy']:>6.4f} | {r['pers_f1']:>6.4f}")

    plt.figure(figsize=(12, 6))
    for r in all_results:
        h = pd.DataFrame(r['history'])
        plt.plot(h['round'], h['accuracy'], label=r['run'], linewidth=1.5)
    plt.title('FedAvg vs Fed-CMA: N-BaIoT Convergence')
    plt.xlabel('Round'); plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7)
    plt.savefig('nbaiot_comparison.png', dpi=150, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
