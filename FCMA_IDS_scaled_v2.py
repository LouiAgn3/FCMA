# ==============================================================================
# FCMA_IDS_scaled_v2.py
# Fixed: LSTM-aware matched averaging + correct evaluation
# ==============================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

print("Libraries imported successfully.")

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
NUM_CLIENTS = 20
NUM_ROUNDS = 100
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SEED = 42

GRADIENT_CLIP_NORM = 1.0
MIN_LR = 0.0001

RECLUSTERING_INTERVAL = 20
LOW_RANK_DIM = 10
SIMILARITY_THRESHOLD = 0.1

# Similarity weights (alpha + beta + gamma = 1.0)
ALPHA = 0.25
BETA = 0.55
GAMMA = 0.20
MODEL_BLEND_WEIGHT = 0.5

# Context sub-weights
W_GEO = 0.4
W_VEHICLE = 0.3
W_ATTACK = 0.3

POISONING_RATE = 0.0
POISONING_FLIP_RATE = 1.0

SEQUENCE_LENGTH = 20
PREPROCESSED_DATA_FILE = 'preprocessed_paper_features_can_data.csv'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.set_per_process_memory_fraction(0.8)


def get_num_clusters(num_clients):
    return max(2, min(10, int(np.sqrt(num_clients))))


# ==============================================================================
# --- METADATA & CONTEXT ---
# ==============================================================================
# Real attack types from the Car Hacking dataset (Kang et al., 2021)
ATTACK_TYPES_IN_DATASET = ['Flooding', 'Fuzzing', 'Replay', 'Spoofing']

# Region and vehicle type remain simulated (SUMO proxy) since the Car Hacking
# dataset was collected from a single Kia Soul. In a real deployment these would
# come from the OBD-II or V2X stack.
REGIONS = ['Urban', 'Highway', 'Rural', 'Suburban']
VEHICLE_TYPES = ['Sedan', 'SUV', 'Truck', 'Compact']


def derive_client_metadata(num_clients, client_attack_profiles, seed=42):
    """
    Builds per-client metadata combining:
      - attack_exposure: REAL, derived from the actual data partition
      - region, vehicle_type: SIMULATED (SUMO proxy), assigned deterministically
        so that clients with similar ECU/attack profiles get correlated context.

    The key insight: clients whose CAN IDs expose them to similar attack types
    (e.g. both see Flooding + Fuzzing) should have correlated simulated context,
    because in a real IVN deployment, ECUs in the same vehicle zone tend to face
    the same threat landscape.
    """
    rng = np.random.default_rng(seed)
    metadata = []

    # Assign region/vehicle based on attack profile similarity to create
    # realistic correlation between context and data distribution
    for i in range(num_clients):
        real_attacks = client_attack_profiles[i]  # Set of actual attack types

        # Heuristic: assign region based on dominant attack pattern
        # This creates meaningful correlation between context and data
        if 'Flooding' in real_attacks and 'Fuzzing' in real_attacks:
            region = 'Urban'        # Dense traffic -> more DoS/fuzzing
            vtype = rng.choice(['Sedan', 'Compact'])
        elif 'Spoofing' in real_attacks and 'Replay' in real_attacks:
            region = 'Highway'      # Long routes -> spoofing/replay
            vtype = rng.choice(['Truck', 'SUV'])
        elif 'Spoofing' in real_attacks or 'Replay' in real_attacks:
            region = 'Suburban'
            vtype = rng.choice(['SUV', 'Sedan'])
        elif 'Flooding' in real_attacks or 'Fuzzing' in real_attacks:
            region = rng.choice(['Urban', 'Suburban'])
            vtype = rng.choice(['Sedan', 'Compact'])
        else:
            # No attacks or uncommon mix
            region = rng.choice(REGIONS)
            vtype = rng.choice(VEHICLE_TYPES)

        metadata.append({
            'client_id': i,
            'region': region,
            'vehicle_type': vtype,
            'attack_exposure': real_attacks,   # REAL from data
            'is_poisoned': False,
        })

    # Log the derived profiles
    print(f"\n--- Derived Client Context Metadata ---")
    for m in metadata:
        atk_str = ', '.join(sorted(m['attack_exposure'])) if m['attack_exposure'] else 'None'
        print(f"  Client {m['client_id']:3d}: {m['region']:>8s} {m['vehicle_type']:>7s} | Attacks: {atk_str}")
    print()

    return metadata


def calculate_s_context(client_metadata, w_geo=W_GEO, w_vehicle=W_VEHICLE, w_attack=W_ATTACK):
    """
    Contextual similarity (Equation 2.7 from thesis).
    S_context(i,j) = w_geo * S_geo + w_vehicle * S_vehicle + w_attack * S_attack

    S_attack now uses Jaccard similarity over REAL attack types
    (Flooding, Fuzzing, Replay, Spoofing) derived from the actual data partition.
    """
    n = len(client_metadata)
    s = np.zeros((n, n))
    for i in range(n):
        s[i, i] = 1.0
        for j in range(i + 1, n):
            mi, mj = client_metadata[i], client_metadata[j]
            # S_geo: indicator (Eq 2.4)
            geo = 1.0 if mi['region'] == mj['region'] else 0.0
            # S_vehicle: indicator (Eq 2.5)
            veh = 1.0 if mi['vehicle_type'] == mj['vehicle_type'] else 0.0
            # S_attack: Jaccard similarity (Eq 2.6)
            si, sj = mi['attack_exposure'], mj['attack_exposure']
            if not si and not sj:
                att = 1.0   # Both clean -> perfectly similar
            else:
                union = si | sj
                att = len(si & sj) / len(union) if union else 1.0
            sim = w_geo * geo + w_vehicle * veh + w_attack * att
            s[i, j] = s[j, i] = sim
    return s


# ==============================================================================
# --- POISONING ---
# ==============================================================================
def apply_label_flipping(client_dataloaders, client_metadata, poisoning_rate, flip_rate=1.0, seed=42):
    rng = np.random.default_rng(seed)
    num_to_poison = max(0, int(len(client_dataloaders) * poisoning_rate))
    if num_to_poison == 0:
        return set()
    eligible = [i for i, l in enumerate(client_dataloaders)
                if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0]
    poisoned_ids = set(rng.choice(eligible, size=min(num_to_poison, len(eligible)), replace=False))
    print(f"\n--- POISONING: {len(poisoned_ids)} clients: {sorted(poisoned_ids)} ---")
    for i, loader in enumerate(client_dataloaders):
        if i in poisoned_ids:
            client_metadata[i]['is_poisoned'] = True
            labels = loader.dataset.labels
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
        'isolated': isolated, 'total_poisoned': len(poisoned_ids),
        'contaminated': contaminated, 'total_clean': len(clean_ids),
    }


# ==============================================================================
# --- DATA ---
# ==============================================================================
def load_preprocessed_data():
    """
    Loads the CSV and preserves the original string labels ('Flooding', 'Fuzzing',
    'Replay', 'Spoofing', 'Normal') in an 'OriginalLabel' column before converting
    to binary for training. This allows us to derive real attack exposure metadata.
    """
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        raise FileNotFoundError(f"'{PREPROCESSED_DATA_FILE}' not found.")
    print(f"Loading data from '{PREPROCESSED_DATA_FILE}'...")
    df = pd.read_csv(PREPROCESSED_DATA_FILE)

    # Preserve original attack types BEFORE binarizing
    df['OriginalLabel'] = df['Label'].copy()
    df['Label'] = (df['Label'] != 'Normal').astype(int)

    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID', 'OriginalLabel']]
    return df[['Arbitration_ID', 'Label', 'OriginalLabel'] + feature_cols]


# ==============================================================================
# --- MODEL ---
# ==============================================================================
class IDS_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(IDS_LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, (h_n, _) = self.lstm2(x)
        x = self.dropout2(h_n.squeeze(0))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CANDataset(Dataset):
    def __init__(self, features, labels, sequence_length):
        self.sequence_length = sequence_length
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0] - self.sequence_length + 1

    def __getitem__(self, idx):
        features_seq = self.features[idx:idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length - 1]
        return features_seq, label.unsqueeze(-1)


# ==============================================================================
# --- HELPERS ---
# ==============================================================================
def partition_data_by_can_id(df, num_clients):
    """
    Partitions data by CAN Arbitration ID and also extracts the real attack
    exposure profile for each client from the original (pre-binarized) labels.
    Returns: train_loaders, test_loaders, client_attack_profiles
    """
    print(f"Partitioning data for {num_clients} clients by CAN ID...")
    can_ids = df['Arbitration_ID'].unique()
    np.random.shuffle(can_ids)
    client_id_map = {can_id: i % num_clients for i, can_id in enumerate(can_ids)}
    df['client_id'] = df['Arbitration_ID'].map(client_id_map)

    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID', 'client_id', 'OriginalLabel']]
    train_loaders, test_loaders = [], []
    client_attack_profiles = []  # Real attack types seen by each client

    for i in range(num_clients):
        client_df = df[df['client_id'] == i].drop(columns=['client_id'])

        # Extract the real attack types this client has been exposed to
        attack_types_seen = set(
            client_df.loc[client_df['OriginalLabel'] != 'Normal', 'OriginalLabel'].unique()
        )
        client_attack_profiles.append(attack_types_seen)

        # Drop OriginalLabel before creating datasets (not a feature)
        client_df = client_df.drop(columns=['OriginalLabel'])

        if len(client_df) < SEQUENCE_LENGTH * 2:
            train_loaders.append(DataLoader([], batch_size=BATCH_SIZE))
            test_loaders.append(DataLoader([], batch_size=BATCH_SIZE))
            continue
        try:
            tr, te = train_test_split(client_df, test_size=0.2, random_state=SEED, stratify=client_df['Label'])
        except ValueError:
            tr, te = train_test_split(client_df, test_size=0.2, random_state=SEED)
        X_tr = StandardScaler().fit_transform(tr[feature_cols].values)
        ds_tr = CANDataset(X_tr, tr['Label'].values, SEQUENCE_LENGTH)
        train_loaders.append(DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True))
        X_te = StandardScaler().fit_transform(te[feature_cols].values)
        ds_te = CANDataset(X_te, te['Label'].values, SEQUENCE_LENGTH)
        test_loaders.append(DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False))

    active = sum(1 for l in train_loaders if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0)
    print(f"  {active}/{num_clients} clients have data.")
    return train_loaders, test_loaders, client_attack_profiles


def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def calculate_model_size(model):
    p = sum(param.nelement() * param.element_size() for param in model.parameters())
    b = sum(buf.nelement() * buf.element_size() for buf in model.buffers())
    return (p + b) / 1024**2


def calculate_s_data(client_dataloaders):
    hists = []
    for loader in client_dataloaders:
        if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0:
            hists.append(np.array([0.5, 0.5]))
            continue
        labels = loader.dataset.labels.numpy()
        total = len(labels)
        if total == 0:
            hists.append(np.array([0.5, 0.5]))
            continue
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
    if not test_loader.dataset or len(test_loader.dataset) == 0:
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
def federated_averaging(models):
    if not models:
        return None
    avg_sd = copy.deepcopy(models[0].state_dict())
    for key in avg_sd:
        avg_sd[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(0)
    agg = IDS_LSTM(models[0].lstm1.input_size).to(DEVICE)
    agg.load_state_dict(avg_sd)
    return agg


def intra_cluster_fedma_lstm(cluster_models, ref_model, threshold):
    """
    LSTM-aware Federated Matched Averaging following the FedMA paper (Wang et al., 2020).

    Key corrections from the previous version:
    ============================================

    1. LSTM GATE-AWARE MATCHING (FedMA Paper Section 2.2):
       PyTorch LSTM packs 4 gate weight matrices (input, forget, cell, output) into
       single tensors: weight_ih = [W_ii; W_if; W_ig; W_io] of shape (4*hidden, input).
       The permutation invariance is over HIDDEN STATES, not individual gate rows.
       We must find ONE permutation for all 4 gates and apply it consistently.

       The paper says: "we stack input-to-hidden weights into SD x L weight matrix
       (S is the number of cell states)" and compute permutation from that.

       So we: stack all 4 gate slices of weight_ih → (4*input_dim, hidden) matrix,
       match hidden units using Hungarian on this stacked representation,
       then apply the SAME permutation to all ih weights, hh weights, and biases.

    2. HIDDEN-TO-HIDDEN CONSISTENCY (FedMA Paper Equation 6):
       For h_t = sigma(h_{t-1} * Pi^T * H * Pi + x_t * W * Pi), the same permutation
       Pi must be applied to BOTH rows and columns of hidden-to-hidden weights.
       Previous code treated hh weights independently, breaking this constraint.

    3. CROSS-LAYER PROPAGATION:
       The permutation found for lstm1's hidden states must be applied to lstm2's
       input dimension, since lstm1's output feeds into lstm2's input.
       Similarly, lstm2's permutation propagates to fc1's input dimension.
    """
    if not cluster_models:
        return None
    if len(cluster_models) == 1:
        return copy.deepcopy(cluster_models[0])

    ref_sd = ref_model.state_dict()
    n_models = len(cluster_models)
    all_sds = [m.state_dict() for m in cluster_models]

    # We'll build the aggregated state dict
    agg_sd = {}

    # ================================================================
    # PHASE 1: Match LSTM1 hidden units (64 hidden units)
    # ================================================================
    # Stack input-to-hidden weights across all 4 gates for matching
    # weight_ih_l0 shape: (4*64, input_dim) = (256, input_dim)
    # We reshape to (64, 4*input_dim) so each row = one hidden unit's full signature

    h1 = 64  # lstm1 hidden size
    input_dim = ref_model.lstm1.input_size

    # Find permutation for each client model relative to reference
    perms_lstm1 = []  # permutation indices for each client

    ref_ih1 = ref_sd['lstm1.weight_ih_l0']  # (256, input_dim)
    # Reshape: split into 4 gates, each (64, input_dim), then concat along dim=1
    # Result: (64, 4*input_dim) - each row is a hidden unit's full gate signature
    ref_gates_ih1 = ref_ih1.view(4, h1, input_dim).permute(1, 0, 2).reshape(h1, 4 * input_dim)

    for m_idx in range(n_models):
        client_ih1 = all_sds[m_idx]['lstm1.weight_ih_l0']
        client_gates_ih1 = client_ih1.view(4, h1, input_dim).permute(1, 0, 2).reshape(h1, 4 * input_dim)

        # Cost matrix: cosine distance between hidden unit signatures
        cost = 1 - torch.nn.functional.cosine_similarity(
            ref_gates_ih1.unsqueeze(1), client_gates_ih1.unsqueeze(0), dim=2
        )
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        # col_ind[ref_idx] = client_idx that matches ref hidden unit ref_idx
        perms_lstm1.append(col_ind)

    # Now aggregate lstm1 with permutations applied
    # weight_ih_l0: (4*h1, input_dim) - permute the hidden unit dimension
    sum_ih1 = torch.zeros_like(ref_sd['lstm1.weight_ih_l0']).float()
    sum_hh1 = torch.zeros_like(ref_sd['lstm1.weight_hh_l0']).float()
    sum_bih1 = torch.zeros_like(ref_sd['lstm1.bias_ih_l0']).float()
    sum_bhh1 = torch.zeros_like(ref_sd['lstm1.bias_hh_l0']).float()

    for m_idx in range(n_models):
        perm = perms_lstm1[m_idx]  # maps ref hidden idx -> client hidden idx
        inv_perm = np.argsort(perm)  # maps client hidden idx -> ref hidden idx

        sd = all_sds[m_idx]

        # weight_ih: (4*h1, input) -> reshape to (4, h1, input), permute h1 dim, reshape back
        w_ih = sd['lstm1.weight_ih_l0'].view(4, h1, input_dim)
        w_ih_permuted = w_ih[:, inv_perm, :]  # reorder client hidden units to match ref
        sum_ih1 += w_ih_permuted.reshape(4 * h1, input_dim)

        # weight_hh: (4*h1, h1) -> permute BOTH dims (Equation 6 from paper)
        w_hh = sd['lstm1.weight_hh_l0'].view(4, h1, h1)
        w_hh_permuted = w_hh[:, inv_perm, :][:, :, inv_perm]  # rows AND cols
        sum_hh1 += w_hh_permuted.reshape(4 * h1, h1)

        # biases: (4*h1,) -> reshape to (4, h1), permute, reshape back
        b_ih = sd['lstm1.bias_ih_l0'].view(4, h1)
        sum_bih1 += b_ih[:, inv_perm].reshape(4 * h1)

        b_hh = sd['lstm1.bias_hh_l0'].view(4, h1)
        sum_bhh1 += b_hh[:, inv_perm].reshape(4 * h1)

    agg_sd['lstm1.weight_ih_l0'] = sum_ih1 / n_models
    agg_sd['lstm1.weight_hh_l0'] = sum_hh1 / n_models
    agg_sd['lstm1.bias_ih_l0'] = sum_bih1 / n_models
    agg_sd['lstm1.bias_hh_l0'] = sum_bhh1 / n_models

    # ================================================================
    # PHASE 2: Match LSTM2 hidden units (32 hidden units)
    # The INPUT to lstm2 comes from lstm1's hidden states, which we just permuted.
    # So we must also permute lstm2's input dimension according to lstm1's permutation.
    # ================================================================

    h2 = 32  # lstm2 hidden size
    h1_out = h1  # lstm2 input = lstm1 hidden size

    # First, apply lstm1 permutation to lstm2's input dimension for each client
    ref_ih2 = ref_sd['lstm2.weight_ih_l0']  # (4*h2, h1)
    ref_gates_ih2 = ref_ih2.view(4, h2, h1_out).permute(1, 0, 2).reshape(h2, 4 * h1_out)

    perms_lstm2 = []
    for m_idx in range(n_models):
        client_ih2 = all_sds[m_idx]['lstm2.weight_ih_l0']  # (4*h2, h1)
        # Apply lstm1's permutation to the input dimension of lstm2
        perm1 = perms_lstm1[m_idx]
        inv_perm1 = np.argsort(perm1)

        client_ih2_aligned = client_ih2.view(4, h2, h1_out)[:, :, inv_perm1].reshape(4 * h2, h1_out)
        client_gates_ih2 = client_ih2_aligned.view(4, h2, h1_out).permute(1, 0, 2).reshape(h2, 4 * h1_out)

        cost = 1 - torch.nn.functional.cosine_similarity(
            ref_gates_ih2.unsqueeze(1), client_gates_ih2.unsqueeze(0), dim=2
        )
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        perms_lstm2.append(col_ind)

    sum_ih2 = torch.zeros_like(ref_sd['lstm2.weight_ih_l0']).float()
    sum_hh2 = torch.zeros_like(ref_sd['lstm2.weight_hh_l0']).float()
    sum_bih2 = torch.zeros_like(ref_sd['lstm2.bias_ih_l0']).float()
    sum_bhh2 = torch.zeros_like(ref_sd['lstm2.bias_hh_l0']).float()

    for m_idx in range(n_models):
        perm1 = perms_lstm1[m_idx]
        inv_perm1 = np.argsort(perm1)
        perm2 = perms_lstm2[m_idx]
        inv_perm2 = np.argsort(perm2)

        sd = all_sds[m_idx]

        # weight_ih: (4*h2, h1) -> first permute input dim (from lstm1), then hidden dim
        w_ih = sd['lstm2.weight_ih_l0'].view(4, h2, h1_out)
        w_ih = w_ih[:, :, inv_perm1]  # align input dimension
        w_ih = w_ih[:, inv_perm2, :]  # align hidden dimension
        sum_ih2 += w_ih.reshape(4 * h2, h1_out)

        # weight_hh: (4*h2, h2) -> permute both dims with perm2
        w_hh = sd['lstm2.weight_hh_l0'].view(4, h2, h2)
        w_hh = w_hh[:, inv_perm2, :][:, :, inv_perm2]
        sum_hh2 += w_hh.reshape(4 * h2, h2)

        b_ih = sd['lstm2.bias_ih_l0'].view(4, h2)
        sum_bih2 += b_ih[:, inv_perm2].reshape(4 * h2)

        b_hh = sd['lstm2.bias_hh_l0'].view(4, h2)
        sum_bhh2 += b_hh[:, inv_perm2].reshape(4 * h2)

    agg_sd['lstm2.weight_ih_l0'] = sum_ih2 / n_models
    agg_sd['lstm2.weight_hh_l0'] = sum_hh2 / n_models
    agg_sd['lstm2.bias_ih_l0'] = sum_bih2 / n_models
    agg_sd['lstm2.bias_hh_l0'] = sum_bhh2 / n_models

    # ================================================================
    # PHASE 3: FC layers — permute fc1 input dim according to lstm2 perm
    # ================================================================
    # fc1.weight: (32, 32) - input dim = lstm2 hidden size
    sum_fc1_w = torch.zeros_like(ref_sd['fc1.weight']).float()
    sum_fc1_b = torch.zeros_like(ref_sd['fc1.bias']).float()

    for m_idx in range(n_models):
        inv_perm2 = np.argsort(perms_lstm2[m_idx])
        sd = all_sds[m_idx]
        # Permute input dimension (columns) of fc1 to match lstm2's aligned hidden states
        sum_fc1_w += sd['fc1.weight'][:, inv_perm2]
        sum_fc1_b += sd['fc1.bias']

    agg_sd['fc1.weight'] = sum_fc1_w / n_models
    agg_sd['fc1.bias'] = sum_fc1_b / n_models

    # fc2 is the final output layer — no permutation needed, just average
    # (FedMA paper: last layer uses weighted average based on class proportions)
    sum_fc2_w = torch.zeros_like(ref_sd['fc2.weight']).float()
    sum_fc2_b = torch.zeros_like(ref_sd['fc2.bias']).float()
    for m_idx in range(n_models):
        sum_fc2_w += all_sds[m_idx]['fc2.weight']
        sum_fc2_b += all_sds[m_idx]['fc2.bias']
    agg_sd['fc2.weight'] = sum_fc2_w / n_models
    agg_sd['fc2.bias'] = sum_fc2_b / n_models

    agg = IDS_LSTM(ref_model.lstm1.input_size).to(DEVICE)
    agg.load_state_dict(agg_sd)
    return agg


# ==============================================================================
# --- MAIN EXPERIMENT ---
# ==============================================================================
def run_experiment(federated_mode, num_clients, use_context=True, poisoning_rate=0.0):
    num_clusters = get_num_clusters(num_clients)

    if federated_mode == 'FCMA' and use_context:
        a_eff, b_eff, g_eff = ALPHA, BETA, GAMMA
    elif federated_mode == 'FCMA' and not use_context:
        a_eff = ALPHA / (ALPHA + BETA)
        b_eff = BETA / (ALPHA + BETA)
        g_eff = 0.0
    else:
        a_eff, b_eff, g_eff = ALPHA, BETA, 0.0

    ctx_label = "_ctx" if (federated_mode == 'FCMA' and use_context) else ""
    poison_label = f"_p{int(poisoning_rate*100)}" if poisoning_rate > 0 else ""
    run_label = f"{federated_mode}_N{num_clients}{ctx_label}{poison_label}"

    print(f"\n{'='*60}")
    print(f"  {run_label}  |  Clusters: {num_clusters}")
    if federated_mode == 'FCMA':
        print(f"  Weights: α={a_eff:.3f}  β={b_eff:.3f}  γ={g_eff:.3f}")
    print(f"  Poisoning: {poisoning_rate*100:.0f}%")
    print(f"{'='*60}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"results_{run_label}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)

    df = load_preprocessed_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['Label'])
    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID', 'OriginalLabel']]
    input_dim = len(feature_cols)

    # Drop OriginalLabel from test set before scaling (it's not a feature)
    X_test = StandardScaler().fit_transform(test_df[feature_cols].values)
    global_test_loader = DataLoader(
        CANDataset(X_test, test_df['Label'].values, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)

    # Partition returns real attack profiles per client
    client_dataloaders, client_test_loaders, client_attack_profiles = partition_data_by_can_id(train_df, num_clients)

    # Derive metadata from REAL data partition (attack exposure from actual labels)
    client_metadata = derive_client_metadata(num_clients, client_attack_profiles, seed=SEED)
    poisoned_ids = apply_label_flipping(
        client_dataloaders, client_metadata, poisoning_rate, POISONING_FLIP_RATE, SEED)

    label_counts = train_df['Label'].value_counts()
    pos_weight = label_counts.get(0, 1) / label_counts.get(1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))

    dist_report = []
    print(f"\n--- Client Data Distribution ({num_clients} clients) ---")
    for cid, loader in enumerate(client_dataloaders):
        if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0:
            dist_report.append(f"Client {cid:3d}: No data")
            continue
        labels = loader.dataset.labels.numpy()
        total = len(labels)
        att = int(np.sum(labels))
        atk_types = ', '.join(sorted(client_attack_profiles[cid])) if client_attack_profiles[cid] else 'None'
        tag = " [POISONED]" if cid in poisoned_ids else ""
        line = f"Client {cid:3d}: {total:7d} samples | Attack: {att/total*100:5.1f}% | Types: {atk_types}{tag}"
        dist_report.append(line)
        if cid < 10 or cid in poisoned_ids:
            print(f"  {line}")
    if num_clients > 10:
        print(f"  ... ({num_clients - 10} more clients, see summary.txt)")
    print("-" * 50)

    # --- Model init ---
    if federated_mode == 'FedAvg':
        global_model = IDS_LSTM(input_dim).to(DEVICE)
        client_cluster_assignments = np.zeros(num_clients, dtype=int)
    else:  # FCMA
        cluster_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(num_clusters)]
        data_sim = calculate_s_data(client_dataloaders)
        if g_eff > 0:
            ctx_sim = calculate_s_context(client_metadata)
            init_sim = (b_eff / (b_eff + g_eff)) * data_sim + (g_eff / (b_eff + g_eff)) * ctx_sim
        else:
            init_sim = data_sim
        dist_mat = 1 - init_sim
        client_cluster_assignments = AgglomerativeClustering(
            n_clusters=num_clusters, metric='precomputed', linkage='average'
        ).fit_predict(dist_mat)
        print(f"Initial cluster assignments: {client_cluster_assignments.tolist()}")

    local_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(num_clients)]

    single_model_mb = calculate_model_size(local_models[0])
    print(f"Model size: {single_model_mb:.2f} MB")
    total_comm_mb = 0

    performance_history = []
    aggregation_times = []
    isolation_history = []

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    for round_num in tqdm(range(NUM_ROUNDS), desc=run_label):

        # --- Phase 2: Re-clustering (FCMA only) ---
        if federated_mode == 'FCMA' and round_num >= 3 and round_num % RECLUSTERING_INTERVAL == 0:
            prev_assign = client_cluster_assignments.copy()
            updates = []
            for cid, model in enumerate(local_models):
                prev_cm = cluster_models[prev_assign[cid]]
                upd = get_flat_params(model) - get_flat_params(prev_cm)
                updates.append(upd.cpu().numpy())
            updates = np.array(updates)

            active_mask = ~np.all(updates == 0, axis=1)
            n_active = np.sum(active_mask)
            M_proj = None
            if n_active > LOW_RANK_DIM:
                pca = PCA(n_components=LOW_RANK_DIM, random_state=SEED)
                pca.fit(updates[active_mask])
                M_proj = pca.components_.T
            elif updates.shape[0] > LOW_RANK_DIM:
                pca = PCA(n_components=LOW_RANK_DIM, random_state=SEED)
                pca.fit(updates)
                M_proj = pca.components_.T

            if M_proj is not None:
                s_model = calculate_s_model(updates, M_proj)
                s_data = calculate_s_data(client_dataloaders)
                combined = a_eff * s_model + b_eff * s_data
                if g_eff > 0:
                    combined += g_eff * calculate_s_context(client_metadata)
                dist_mat = 1 - combined

                client_cluster_assignments = AgglomerativeClustering(
                    n_clusters=num_clusters, metric='precomputed', linkage='average'
                ).fit_predict(dist_mat)

                tqdm.write(f"  Round {round_num+1}: Re-clustered → {client_cluster_assignments.tolist()}")

                if poisoned_ids:
                    iso = calculate_poisoning_isolation_rate(
                        client_cluster_assignments, poisoned_ids, num_clients)
                    isolation_history.append({'round': round_num + 1, **iso})

        # --- Phase 1: Local Training ---
        current_local = []
        n_active_round = 0
        for cid, loader in enumerate(client_dataloaders):
            if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0 or len(loader.dataset) == 0:
                current_local.append(copy.deepcopy(local_models[cid]))
                continue

            n_active_round += 1

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
                        m = IDS_LSTM(input_dim).to(DEVICE)
                        m.load_state_dict(blended)
                    else:
                        m = copy.deepcopy(cluster_models[cidx])
                else:
                    m = copy.deepcopy(cluster_models[cidx])

            m.lstm1.flatten_parameters()
            m.lstm2.flatten_parameters()
            lr = max(LEARNING_RATE * (0.95 ** (round_num // 10)), MIN_LR)
            optimizer = optim.Adam(m.parameters(), lr=lr)
            m.train()

            if torch.cuda.is_available() and round_num % 5 == 0:
                torch.cuda.empty_cache()

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

        if len(current_local) == num_clients:
            local_models = current_local

        total_comm_mb += 2 * n_active_round * single_model_mb

        # --- Phase 3: Aggregation ---
        t0 = time.time()
        active_models = [local_models[i] for i, l in enumerate(client_dataloaders)
                         if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0]

        if federated_mode == 'FedAvg':
            if active_models:
                global_model = federated_averaging(active_models)
        else:  # FCMA
            for cl_id in range(num_clusters):
                models_in = [local_models[i] for i, c in enumerate(client_cluster_assignments)
                             if c == cl_id and i < len(local_models) and
                             hasattr(client_dataloaders[i].dataset, 'labels') and
                             len(client_dataloaders[i].dataset.labels) > 0]
                if models_in:
                    agg = intra_cluster_fedma_lstm(
                        models_in, cluster_models[cl_id], SIMILARITY_THRESHOLD)
                    if agg:
                        cluster_models[cl_id] = agg

        aggregation_times.append(time.time() - t0)

        # --- Evaluate ---
        # FIX: For FCMA, evaluate each client with ITS OWN cluster model on its local
        # test set (personalized accuracy). For global tracking, use the best/largest
        # cluster model or a weighted vote. We also evaluate each cluster on global
        # for comparison.
        if federated_mode == 'FedAvg':
            acc, pre, rec, f1 = evaluate(global_model, global_test_loader, return_metrics=True)
        else:
            # Personalized evaluation: each client uses its cluster's model on its local test
            round_accs = []
            round_f1s = []
            for cid in range(num_clients):
                tl = client_test_loaders[cid]
                if not hasattr(tl.dataset, 'labels') or len(tl.dataset) == 0:
                    continue
                cl = client_cluster_assignments[cid]
                a, p, r, f = evaluate(cluster_models[cl], tl, return_metrics=True)
                round_accs.append(a)
                round_f1s.append(f)
            acc = np.mean(round_accs) if round_accs else 0.0
            f1 = np.mean(round_f1s) if round_f1s else 0.0
            pre, rec = 0.0, 0.0  # simplified for round logging

        if (round_num + 1) % 10 == 0 or round_num == 0:
            tqdm.write(f"  Round {round_num+1:3d}: Acc={acc:.4f}  F1={f1:.4f}  Comm={total_comm_mb:.1f}MB")

        performance_history.append({
            'round': round_num + 1, 'accuracy': acc, 'f1': f1,
            'precision': pre, 'recall': rec})

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print(f"\n--- Final Evaluation: {run_label} ---")
    print(f"Avg aggregation time: {np.mean(aggregation_times):.4f}s/round")
    print(f"Total communication: {total_comm_mb:.1f} MB")

    # Personalization: each client evaluated with its cluster model on its local test
    local_accs, local_f1s = [], []
    clean_accs, clean_f1s = [], []
    pers_lines = []
    for cid in range(num_clients):
        tl = client_test_loaders[cid]
        if not hasattr(tl.dataset, 'labels') or len(tl.dataset) == 0:
            continue
        if federated_mode == 'FedAvg':
            m_eval = global_model
            cl_str = "N/A"
        else:
            cl = client_cluster_assignments[cid]
            m_eval = cluster_models[cl]
            cl_str = str(cl)

        a, p, r, f = evaluate(m_eval, tl, return_metrics=True)
        local_accs.append(a)
        local_f1s.append(f)
        tag = " [P]" if cid in poisoned_ids else ""
        if cid not in poisoned_ids:
            clean_accs.append(a)
            clean_f1s.append(f)
        line = f"Client {cid:3d} (Cl {cl_str:>3s}): Acc={a:.4f} F1={f:.4f}{tag}"
        pers_lines.append(line)

    avg_acc = np.mean(local_accs) if local_accs else 0
    avg_f1 = np.mean(local_f1s) if local_f1s else 0
    avg_clean_acc = np.mean(clean_accs) if clean_accs else 0
    avg_clean_f1 = np.mean(clean_f1s) if clean_f1s else 0

    print(f"\n  Personalized Accuracy:  {avg_acc:.4f}")
    print(f"  Personalized F1-Score:  {avg_f1:.4f}")
    if poisoned_ids:
        print(f"  Clean Client Accuracy:  {avg_clean_acc:.4f}")

    # --- Also report global test accuracy for fair comparison with FedAvg ---
    if federated_mode == 'FCMA':
        # Use majority-vote or best cluster on global test
        best_global_acc = 0
        for m in cluster_models:
            ga = evaluate(m, global_test_loader)
            best_global_acc = max(best_global_acc, ga)
        print(f"  Best cluster global acc: {best_global_acc:.4f}")

    # --- Plots ---
    hist_df = pd.DataFrame(performance_history)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_df['round'], hist_df['accuracy'], 'o-', label='Accuracy', markersize=2)
    plt.plot(hist_df['round'], hist_df['f1'], 'x--', label='F1-Score', markersize=2)
    plt.title(f'Convergence: {run_label}')
    plt.xlabel('Round')
    plt.ylabel('Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_folder, 'convergence.png'), dpi=150)
    plt.close()

    final_model = global_model if federated_mode == 'FedAvg' else cluster_models[0]
    y_true, y_pred = evaluate(final_model, global_test_loader, return_preds=True)
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix: {run_label}')
        plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'), dpi=150)
        plt.close()

    # --- Summary ---
    with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
        f.write(f"=== {run_label} ===\n\n")
        f.write(f"Clients: {num_clients}\nClusters: {num_clusters}\nRounds: {NUM_ROUNDS}\n")
        f.write(f"Mode: {federated_mode}\nContext: {use_context}\nPoisoning: {poisoning_rate*100:.0f}%\n")
        if federated_mode == 'FCMA':
            f.write(f"Weights: α={a_eff:.3f} β={b_eff:.3f} γ={g_eff:.3f}\n")
        f.write(f"\n--- Data Distribution ---\n")
        f.write("\n".join(dist_report))
        f.write(f"\n\n--- Communication ---\nTotal: {total_comm_mb:.1f} MB\n")
        f.write(f"\n--- Personalization ---\n")
        f.write("\n".join(pers_lines))
        f.write(f"\n\nAvg Accuracy: {avg_acc:.4f}\nAvg F1: {avg_f1:.4f}\n")

    print(f"Results saved to {results_folder}/")

    return {
        'run': run_label, 'mode': federated_mode, 'num_clients': num_clients,
        'poisoning': poisoning_rate, 'pers_accuracy': avg_acc, 'pers_f1': avg_f1,
        'clean_accuracy': avg_clean_acc, 'history': performance_history,
    }


# ==============================================================================
# --- EXECUTION ---
# ==============================================================================
def main():
    all_results = []

    for n_clients in [20, 50]:
        for mode in ['FedAvg', 'FCMA']:
            print(f"\n{'#'*60}")
            print(f"#  {mode} with {n_clients} clients")
            print(f"{'#'*60}")
            result = run_experiment(
                federated_mode=mode, num_clients=n_clients,
                use_context=True, poisoning_rate=0.0)
            all_results.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':<30s} | {'Pers Acc':>8s} | {'Pers F1':>8s}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['run']:<30s} | {r['pers_accuracy']:>8.4f} | {r['pers_f1']:>8.4f}")

    plt.figure(figsize=(12, 6))
    for r in all_results:
        h = pd.DataFrame(r['history'])
        plt.plot(h['round'], h['accuracy'], label=f"{r['run']}", linewidth=1.5)
    plt.title('FedAvg vs Fed-CMA: Convergence Comparison')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.savefig('comparison_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nComparative plot saved to comparison_convergence.png")


if __name__ == '__main__':
    main()
