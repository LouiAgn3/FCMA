# --- SETUP AND IMPORTS ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
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
from itertools import combinations

print("Libraries imported successfully.")


# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# Federated Learning Hyperparameters
NUM_CLIENTS = 5
NUM_ROUNDS = 100
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SEED = 42

# Additional stability parameters
GRADIENT_CLIP_NORM = 1.0
MIN_LR = 0.0001

# FCMA / FedMA Specific Hyperparameters
NUM_CLUSTERS = 5
RECLUSTERING_INTERVAL = 20
LOW_RANK_DIM = 10
SIMILARITY_THRESHOLD = 0.1

# Similarity Metric Weights (Full Fed-CMA: alpha + beta + gamma = 1.0)
ALPHA = 0.25   # Model update similarity weight
BETA = 0.55    # Data distribution similarity weight
GAMMA = 0.20   # Contextual similarity weight
MODEL_BLEND_WEIGHT = 0.5  # Weight for blending when switching clusters

# Contextual Similarity Sub-Weights (w1 + w2 + w3 = 1.0)
W_GEO = 0.4      # Geographic zone similarity
W_VEHICLE = 0.3   # Vehicle type similarity
W_ATTACK = 0.3    # Attack pattern similarity

# Poisoning Configuration
POISONING_RATE = 0.0       # Fraction of clients to poison (0.0 = no poisoning)
POISONING_FLIP_RATE = 1.0  # Fraction of labels to flip on poisoned clients

# Data and Model Configuration
SEQUENCE_LENGTH = 20
PREPROCESSED_DATA_FILE = 'preprocessed_paper_features_can_data.csv'

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    torch.cuda.set_per_process_memory_fraction(0.8)


# ==============================================================================
# --- SYNTHETIC METADATA GENERATION (Simulating SUMO Outputs) ---
# ==============================================================================

# These constants define the simulation "world" for contextual metadata.
# In a real deployment, this would come from SUMO or real vehicle telemetry.
REGIONS = ['Urban', 'Highway', 'Rural', 'Suburban']
VEHICLE_TYPES = ['Sedan', 'SUV', 'Truck', 'Compact']
ATTACK_TYPES_AVAILABLE = ['DoS', 'Fuzzy', 'Spoofing_Gear', 'Spoofing_RPM']


def generate_client_metadata(num_clients, seed=42):
    """
    Generates synthetic contextual metadata for each client to simulate
    the SUMO outputs described in the thesis methodology (Section 2.2.2).

    Each client receives:
      - region: A geographic zone (e.g., 'Urban', 'Highway')
      - vehicle_type: A vehicle class (e.g., 'Sedan', 'Truck')
      - attack_exposure: A set of attack types the client has been exposed to

    The generation creates *structured* heterogeneity:
      - Clients 0-1: Urban Sedans exposed to DoS/Fuzzy (high-density traffic)
      - Client 2:    Highway Truck exposed to RPM Spoofing (long-haul)
      - Client 3:    Suburban SUV with broad attack exposure
      - Client 4+:   Rural Compacts with minimal exposure (baseline)

    This deliberate structure tests whether S_context can capture meaningful
    real-world correlations between environment and threat landscape.
    """
    rng = np.random.default_rng(seed)
    metadata = []

    for i in range(num_clients):
        if i < num_clients // 3:
            # Urban cluster: high traffic, DoS-heavy
            region = 'Urban'
            vtype = rng.choice(['Sedan', 'Compact'])
            attacks = {'DoS', 'Fuzzy'}
        elif i < 2 * num_clients // 3:
            # Highway cluster: spoofing-heavy
            region = 'Highway'
            vtype = rng.choice(['Truck', 'SUV'])
            attacks = {'Spoofing_RPM', 'Spoofing_Gear'}
        else:
            # Rural/Suburban: mixed or minimal
            region = rng.choice(['Rural', 'Suburban'])
            vtype = rng.choice(['Sedan', 'SUV', 'Compact'])
            # Varied exposure
            n_attacks = rng.integers(0, len(ATTACK_TYPES_AVAILABLE) + 1)
            attacks = set(rng.choice(ATTACK_TYPES_AVAILABLE, size=n_attacks, replace=False)) if n_attacks > 0 else set()

        metadata.append({
            'client_id': i,
            'region': region,
            'vehicle_type': vtype,
            'attack_exposure': attacks,
        })

    return metadata


# ==============================================================================
# --- CONTEXTUAL SIMILARITY METRIC (Thesis Equations 2.4 - 2.7) ---
# ==============================================================================

def calculate_s_geo(meta_i, meta_j):
    """
    Geographic Similarity (Equation 2.4):
      S_geo = I(zone_1 = zone_2)
    Returns 1.0 if clients are in the same region, 0.0 otherwise.
    """
    return 1.0 if meta_i['region'] == meta_j['region'] else 0.0


def calculate_s_vehicle(meta_i, meta_j):
    """
    Vehicle Type Similarity (Equation 2.5):
      S_vehicle = I(type_1 = type_2)
    Returns 1.0 if clients have the same vehicle type, 0.0 otherwise.
    """
    return 1.0 if meta_i['vehicle_type'] == meta_j['vehicle_type'] else 0.0


def calculate_s_attack(meta_i, meta_j):
    """
    Attack Pattern Similarity (Equation 2.6):
      S_attack = |Attacks_1 ∩ Attacks_2| / |Attacks_1 ∪ Attacks_2|
    Jaccard similarity of attack exposure sets.
    Returns 1.0 if both have empty sets (both unexposed = similar).
    """
    set_i = meta_i['attack_exposure']
    set_j = meta_j['attack_exposure']

    if len(set_i) == 0 and len(set_j) == 0:
        return 1.0  # Both unexposed = maximally similar

    union = set_i | set_j
    if len(union) == 0:
        return 1.0

    intersection = set_i & set_j
    return len(intersection) / len(union)


def calculate_s_context(client_metadata, w_geo=W_GEO, w_vehicle=W_VEHICLE, w_attack=W_ATTACK):
    """
    Full Contextual Similarity Matrix (Equation 2.7):
      S_context = w1 * S_geo + w2 * S_vehicle + w3 * S_attack

    Args:
        client_metadata: List of metadata dicts from generate_client_metadata()
        w_geo, w_vehicle, w_attack: Sub-component weights

    Returns:
        np.ndarray: Pairwise contextual similarity matrix (N x N)
    """
    n = len(client_metadata)
    s_context = np.zeros((n, n))

    for i in range(n):
        s_context[i, i] = 1.0  # Self-similarity is always 1
        for j in range(i + 1, n):
            geo = calculate_s_geo(client_metadata[i], client_metadata[j])
            veh = calculate_s_vehicle(client_metadata[i], client_metadata[j])
            att = calculate_s_attack(client_metadata[i], client_metadata[j])

            sim = w_geo * geo + w_vehicle * veh + w_attack * att
            s_context[i, j] = sim
            s_context[j, i] = sim

    return s_context


# ==============================================================================
# --- POISONING ATTACK IMPLEMENTATION ---
# ==============================================================================

def apply_label_flipping(client_dataloaders, client_metadata, poisoning_rate, flip_rate=1.0, seed=42):
    """
    Simulates a label-flipping poisoning attack on a fraction of clients.

    Selects `poisoning_rate` fraction of clients and flips `flip_rate` fraction
    of their labels (0→1 and 1→0). This models a Byzantine adversary that
    attempts to corrupt the global model by injecting incorrect training signals.

    Args:
        client_dataloaders: List of DataLoaders for each client
        client_metadata: List of metadata dicts (will be tagged with 'is_poisoned')
        poisoning_rate: Fraction of clients to poison (0.0 to 1.0)
        flip_rate: Fraction of labels to flip on each poisoned client
        seed: Random seed for reproducibility

    Returns:
        poisoned_client_ids: Set of client IDs that were poisoned
    """
    rng = np.random.default_rng(seed)
    num_to_poison = max(0, int(len(client_dataloaders) * poisoning_rate))

    if num_to_poison == 0:
        # Tag all clients as clean
        for meta in client_metadata:
            meta['is_poisoned'] = False
        return set()

    # Select which clients to poison
    eligible = [i for i, loader in enumerate(client_dataloaders)
                if hasattr(loader.dataset, 'labels') and len(loader.dataset.labels) > 0]
    poisoned_ids = set(rng.choice(eligible, size=min(num_to_poison, len(eligible)), replace=False))

    print(f"\n--- POISONING ATTACK ---")
    print(f"Poisoning {len(poisoned_ids)} of {len(client_dataloaders)} clients: {sorted(poisoned_ids)}")

    for i, loader in enumerate(client_dataloaders):
        if i in poisoned_ids:
            client_metadata[i]['is_poisoned'] = True
            labels = loader.dataset.labels
            n_flip = int(len(labels) * flip_rate)
            flip_indices = rng.choice(len(labels), size=n_flip, replace=False)

            original_attack_count = int(labels.sum().item())
            # Flip: 0→1 and 1→0
            labels[flip_indices] = 1.0 - labels[flip_indices]
            new_attack_count = int(labels.sum().item())

            print(f"  Client {i}: Flipped {n_flip}/{len(labels)} labels "
                  f"(Attack count: {original_attack_count} → {new_attack_count})")
        else:
            client_metadata[i]['is_poisoned'] = False

    return poisoned_ids


def calculate_poisoning_isolation_rate(client_cluster_assignments, poisoned_client_ids, num_clients):
    """
    Measures how effectively the clustering isolates poisoned clients.

    Isolation Rate = fraction of poisoned clients that are in clusters
    containing ONLY poisoned clients (no benign clients corrupted).

    A high rate means the clustering successfully quarantined the adversaries.

    Args:
        client_cluster_assignments: Array of cluster IDs for each client
        poisoned_client_ids: Set of poisoned client IDs
        num_clients: Total number of clients

    Returns:
        dict with isolation metrics
    """
    if not poisoned_client_ids:
        return {'isolation_rate': 1.0, 'contamination_rate': 0.0, 'details': 'No poisoned clients'}

    clean_ids = set(range(num_clients)) - poisoned_client_ids

    # Find which clusters contain poisoned clients
    poisoned_clusters = set()
    for pid in poisoned_client_ids:
        if pid < len(client_cluster_assignments):
            poisoned_clusters.add(client_cluster_assignments[pid])

    # Check if clean clients are in those clusters (contamination)
    contaminated_clean = 0
    for cid in clean_ids:
        if cid < len(client_cluster_assignments):
            if client_cluster_assignments[cid] in poisoned_clusters:
                contaminated_clean += 1

    # Isolation: poisoned clients in clusters with NO clean clients
    isolated_poisoned = 0
    for pid in poisoned_client_ids:
        if pid < len(client_cluster_assignments):
            cluster = client_cluster_assignments[pid]
            # Check if any clean client is in this cluster
            clean_in_cluster = any(
                client_cluster_assignments[c] == cluster for c in clean_ids
                if c < len(client_cluster_assignments)
            )
            if not clean_in_cluster:
                isolated_poisoned += 1

    isolation_rate = isolated_poisoned / len(poisoned_client_ids) if poisoned_client_ids else 1.0
    contamination_rate = contaminated_clean / len(clean_ids) if clean_ids else 0.0

    return {
        'isolation_rate': isolation_rate,
        'contamination_rate': contamination_rate,
        'isolated_poisoned': isolated_poisoned,
        'total_poisoned': len(poisoned_client_ids),
        'contaminated_clean': contaminated_clean,
        'total_clean': len(clean_ids),
    }


# ==============================================================================
# --- DATA LOADING ---
# ==============================================================================

def load_preprocessed_data():
    """Loads the pre-engineered data file and prepares it for the experiment."""
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        raise FileNotFoundError(
            f"Error: The preprocessed data file '{PREPROCESSED_DATA_FILE}' was not found. "
            f"Please ensure the script that generates it has been run successfully."
        )

    print(f"Loading preprocessed data from '{PREPROCESSED_DATA_FILE}'...")
    df = pd.read_csv(PREPROCESSED_DATA_FILE)

    print("Converting labels to binary format (0: Normal, 1: Attack)...")
    df['Label'] = (df['Label'] != 'Normal').astype(int)

    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID']]

    return df[['Arbitration_ID', 'Label'] + feature_cols]


# ==============================================================================
# --- PYTORCH MODEL AND DATASET ---
# ==============================================================================

class IDS_LSTM(nn.Module):
    """PyTorch LSTM model for CAN Intrusion Detection."""
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
    """Custom PyTorch Dataset for CAN bus sequences."""
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
# --- FEDERATED LEARNING HELPERS ---
# ==============================================================================

def partition_data_by_can_id(df, num_clients):
    """
    Partitions data among clients based on Arbitration_ID.
    Each client's data is split into a local training and test set.
    """
    print("Partitioning data by CAN ID and creating local train/test splits...")
    can_ids = df['Arbitration_ID'].unique()
    np.random.shuffle(can_ids)
    client_id_map = {can_id: i % num_clients for i, can_id in enumerate(can_ids)}

    df['client_id'] = df['Arbitration_ID'].map(client_id_map)

    client_train_dataloaders = []
    client_test_dataloaders = []
    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID', 'client_id']]

    for i in range(num_clients):
        client_df = df[df['client_id'] == i].drop(columns=['client_id'])

        if len(client_df) < SEQUENCE_LENGTH * 2:
            client_train_dataloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            client_test_dataloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            continue

        try:
            train_client_df, test_client_df = train_test_split(
                client_df, test_size=0.2, random_state=SEED, stratify=client_df['Label']
            )
        except ValueError:
            train_client_df, test_client_df = train_test_split(
                client_df, test_size=0.2, random_state=SEED
            )

        X_train = train_client_df[feature_cols].values
        y_train = train_client_df['Label'].values
        scaler_train = StandardScaler()
        X_train_scaled = scaler_train.fit_transform(X_train)
        train_dataset = CANDataset(X_train_scaled, y_train, SEQUENCE_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        client_train_dataloaders.append(train_loader)

        X_test = test_client_df[feature_cols].values
        y_test = test_client_df['Label'].values
        scaler_test = StandardScaler()
        X_test_scaled = scaler_test.fit_transform(X_test)
        test_dataset = CANDataset(X_test_scaled, y_test, SEQUENCE_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        client_test_dataloaders.append(test_loader)

    print(f"Data partitioned for {len(client_train_dataloaders)} clients.")
    return client_train_dataloaders, client_test_dataloaders


def calculate_model_size(model):
    """Calculates the size of a PyTorch model in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def get_flat_params(model):
    """Flattens model parameters into a single tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def calculate_s_data(client_dataloaders):
    """Calculates data similarity based on the ratio of Normal/Attack packets."""
    client_histograms = []
    for loader in client_dataloaders:
        if not hasattr(loader.dataset, 'labels'):
            client_histograms.append(np.array([0.5, 0.5]))
            continue
        labels = loader.dataset.labels.numpy()
        if len(labels) == 0:
            client_histograms.append(np.array([0.5, 0.5]))
            continue
        normal_count = np.sum(labels == 0)
        attack_count = np.sum(labels == 1)
        total = len(labels)
        if total == 0:
            client_histograms.append(np.array([0.5, 0.5]))
            continue
        hist = np.array([normal_count / total, attack_count / total])
        client_histograms.append(hist)

    s_data = np.zeros((len(client_histograms), len(client_histograms)))
    for i in range(len(client_histograms)):
        for j in range(i, len(client_histograms)):
            sim = 1 - jensenshannon(client_histograms[i], client_histograms[j])
            s_data[i, j] = s_data[j, i] = sim
    return s_data


def calculate_s_model(model_updates, M):
    """Calculates model similarity based on cosine similarity of low-rank projected updates."""
    projected_updates = model_updates @ M
    norm = np.linalg.norm(projected_updates, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    cosine_sim = (projected_updates @ projected_updates.T) / (norm @ norm.T)
    normalized_sim = np.clip(cosine_sim, 0, 1)
    return normalized_sim


def evaluate(model, test_loader, return_metrics=False, return_preds=False):
    """Evaluates the model's performance."""
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
            all_targets, all_preds, average='binary', zero_division=0
        )
        return accuracy, precision, recall, f1
    return accuracy


# ==============================================================================
# --- AGGREGATION ALGORITHMS ---
# ==============================================================================

def federated_averaging(models):
    """Standard Federated Averaging."""
    if not models:
        return None
    avg_state_dict = copy.deepcopy(models[0].state_dict())
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(0)
    input_dim = models[0].lstm1.input_size
    aggregated_model = IDS_LSTM(input_dim).to(DEVICE)
    aggregated_model.load_state_dict(avg_state_dict)
    return aggregated_model


def intra_cluster_fedma(cluster_models, ref_model, threshold):
    """Federated Matched Averaging for a cluster."""
    if not cluster_models:
        return None

    ref_model_state_dict = copy.deepcopy(ref_model.state_dict())
    aggregated_state_dict = {}

    param_accumulators = {
        name: [m.state_dict()[name].clone() for m in cluster_models]
        for name in ref_model_state_dict.keys()
    }

    for name, ref_params in ref_model_state_dict.items():
        if 'weight' in name and len(ref_params.shape) > 1:
            ref_neurons = ref_params.view(ref_params.size(0), -1)
            sum_neurons = torch.zeros_like(ref_neurons)
            count_neurons = torch.zeros(ref_neurons.size(0), device=DEVICE)

            for other_params in param_accumulators[name]:
                other_neurons = other_params.view(other_params.size(0), -1)
                if ref_neurons.shape[0] != other_neurons.shape[0]:
                    continue
                cost = 1 - torch.nn.functional.cosine_similarity(
                    ref_neurons.unsqueeze(1), other_neurons.unsqueeze(0), dim=2
                )
                row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
                for r, c in zip(row_ind, col_ind):
                    sim = 1 - cost[r, c]
                    if sim >= threshold:
                        sum_neurons[r] += other_neurons[c]
                        count_neurons[r] += 1

            avg_neurons = sum_neurons / count_neurons.clamp(min=1).unsqueeze(1)
            unmatched_indices = torch.where(count_neurons == 0)[0]
            if len(unmatched_indices) > 0:
                avg_neurons[unmatched_indices] = ref_neurons[unmatched_indices]
            aggregated_state_dict[name] = avg_neurons.view_as(ref_params)
        else:
            aggregated_state_dict[name] = torch.stack(param_accumulators[name]).mean(0)

    input_dim = ref_model.lstm1.input_size
    aggregated_model = IDS_LSTM(input_dim).to(DEVICE)
    aggregated_model.load_state_dict(aggregated_state_dict)
    return aggregated_model


# ==============================================================================
# --- MAIN EXPERIMENT LOGIC ---
# ==============================================================================

def run_experiment(FEDERATED_MODE, use_context=True, poisoning_rate=0.0):
    """
    Main function to run the federated learning experiment.

    Args:
        FEDERATED_MODE: One of 'FCMA', 'FedMA', 'FedAvg', 'FCAvg'
        use_context: If True, include S_context in clustering (for ablation)
        poisoning_rate: Fraction of clients to poison (0.0 = no poisoning)
    """
    # --- Derive effective weights based on ablation config ---
    if FEDERATED_MODE == 'FCMA' and use_context:
        alpha_eff, beta_eff, gamma_eff = ALPHA, BETA, GAMMA
        mode_label = f"{FEDERATED_MODE}_full"
    elif FEDERATED_MODE == 'FCMA' and not use_context:
        # Ablation: gamma=0, redistribute weight proportionally
        alpha_eff = ALPHA / (ALPHA + BETA)
        beta_eff = BETA / (ALPHA + BETA)
        gamma_eff = 0.0
        mode_label = f"{FEDERATED_MODE}_no_context"
    else:
        alpha_eff = ALPHA
        beta_eff = BETA
        gamma_eff = 0.0
        mode_label = FEDERATED_MODE

    poison_label = f"_poison{int(poisoning_rate*100)}" if poisoning_rate > 0 else ""

    print(f"\nRunning in mode: {mode_label}{poison_label}")
    if FEDERATED_MODE == 'FCMA':
        print(f"  Effective weights: α={alpha_eff:.3f}, β={beta_eff:.3f}, γ={gamma_eff:.3f}")
    print(f"  Poisoning rate: {poisoning_rate*100:.0f}%")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- Create results folder ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = f"results_{mode_label}{poison_label}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved in: {results_folder}")

    performance_history = []
    aggregation_times = []
    isolation_history = []  # Track poisoning isolation over rounds

    # --- Load Data ---
    df = load_preprocessed_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['Label'])

    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID']]
    X_test = test_df[feature_cols].values
    y_test = test_df['Label'].values
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    global_test_dataset = CANDataset(X_test_scaled, y_test, SEQUENCE_LENGTH)
    global_test_loader = DataLoader(global_test_dataset, batch_size=BATCH_SIZE)

    client_dataloaders, client_test_loaders = partition_data_by_can_id(train_df, NUM_CLIENTS)
    input_dim = len(feature_cols)

    # --- Generate Contextual Metadata ---
    client_metadata = generate_client_metadata(NUM_CLIENTS, seed=SEED)
    print("\n--- Client Contextual Metadata ---")
    for meta in client_metadata:
        print(f"  Client {meta['client_id']}: Region={meta['region']}, "
              f"Type={meta['vehicle_type']}, Attacks={meta['attack_exposure']}")

    # --- Apply Poisoning Attack ---
    poisoned_client_ids = apply_label_flipping(
        client_dataloaders, client_metadata, poisoning_rate,
        flip_rate=POISONING_FLIP_RATE, seed=SEED
    )

    # --- Class Weights ---
    print("\nAccounting for skewed data by calculating class weights...")
    label_counts = train_df['Label'].value_counts()
    num_normal = label_counts.get(0, 1)
    num_attack = label_counts.get(1, 1)
    pos_weight = num_normal / num_attack
    print(f"Normal samples: {num_normal}, Attack samples: {num_attack}")
    print(f"Class weight (pos_weight): {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))

    # --- Model Initialization ---
    if FEDERATED_MODE == 'FedAvg':
        print("Initializing global model for FedAvg...")
        global_model = IDS_LSTM(input_dim).to(DEVICE)
    elif FEDERATED_MODE == 'FedMA':
        print("Initializing single model group for FedMA (Matched Averaging)...")
        cluster_models = [IDS_LSTM(input_dim).to(DEVICE)]
        client_cluster_assignments = np.zeros(NUM_CLIENTS, dtype=int)
    elif FEDERATED_MODE in ['FCMA', 'FCAvg']:
        print(f"Initializing models and clusters for {FEDERATED_MODE}...")
        cluster_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLUSTERS)]
        print("Performing initial clustering based on data similarity...")
        data_sim_matrix = calculate_s_data(client_dataloaders)

        # Include context in initial clustering if enabled
        if gamma_eff > 0:
            context_sim_matrix = calculate_s_context(client_metadata)
            # For initial clustering, use beta + gamma only (no model updates yet)
            init_sim = (beta_eff / (beta_eff + gamma_eff)) * data_sim_matrix + \
                       (gamma_eff / (beta_eff + gamma_eff)) * context_sim_matrix
        else:
            init_sim = data_sim_matrix

        distance_matrix = 1 - init_sim
        clusterer = AgglomerativeClustering(
            n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average'
        )
        client_cluster_assignments = clusterer.fit_predict(distance_matrix)
        print("Initial clustering complete.")

    local_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLIENTS)]

    # --- Capture initial data distribution ---
    initial_data_distribution_report = []
    print("\n--- Client Data Distribution (Training Sets) ---")
    for client_id, loader in enumerate(client_dataloaders):
        if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0:
            line = f"Client {client_id}: No data assigned."
            print(line)
            initial_data_distribution_report.append(line)
            continue
        labels = loader.dataset.labels.numpy()
        total_samples = len(labels)
        attack_count = int(np.sum(labels))
        normal_count = total_samples - attack_count
        if total_samples > 0:
            attack_perc = (attack_count / total_samples) * 100
            normal_perc = (normal_count / total_samples) * 100
            poisoned_tag = " [POISONED]" if client_id in poisoned_client_ids else ""
            line = (
                f"Client {client_id}: {total_samples} samples -> "
                f"Normal: {normal_count} ({normal_perc:.1f}%), "
                f"Attack: {attack_count} ({attack_perc:.1f}%){poisoned_tag}"
            )
            print(line)
            initial_data_distribution_report.append(line)
        else:
            line = f"Client {client_id}: Empty dataset after processing."
            print(line)
            initial_data_distribution_report.append(line)
    print("-" * 50)

    single_model_size_mb = calculate_model_size(local_models[0])
    print(f"\nSize of one model transfer: {single_model_size_mb:.2f} MB")
    total_communication_cost_mb = 0

    # ==========================================
    # --- TRAINING LOOP ---
    # ==========================================
    for round_num in tqdm(range(NUM_ROUNDS), desc=f"Federated Rounds ({mode_label})"):

        # --- Phase 2: Dynamic Re-Clustering ---
        if FEDERATED_MODE in ['FCMA', 'FCAvg', 'FedMA'] and round_num >= 3 and round_num % RECLUSTERING_INTERVAL == 0:
            prev_assignments = client_cluster_assignments.copy()
            model_updates = []
            for client_id, model in enumerate(local_models):
                if FEDERATED_MODE == 'FedMA':
                    prev_cluster_model = cluster_models[0]
                else:
                    prev_cluster_model = cluster_models[prev_assignments[client_id]]
                update = get_flat_params(model) - get_flat_params(prev_cluster_model)
                model_updates.append(update.cpu().numpy())
            model_updates = np.array(model_updates)

            # PCA for low-rank projection
            pca = PCA(n_components=LOW_RANK_DIM, random_state=SEED)
            if np.any(np.all(model_updates == 0, axis=1)):
                active_clients_mask = ~np.all(model_updates == 0, axis=1)
                if np.sum(active_clients_mask) > LOW_RANK_DIM:
                    pca.fit(model_updates[active_clients_mask])
                    M = pca.components_.T
                else:
                    M = np.random.rand(model_updates.shape[1], LOW_RANK_DIM)
            elif model_updates.shape[0] > LOW_RANK_DIM:
                pca.fit(model_updates)
                M = pca.components_.T
            else:
                tqdm.write("--- Skipping re-clustering: Not enough clients for PCA ---")
                continue

            model_sim_matrix = calculate_s_model(model_updates, M)

            if FEDERATED_MODE == 'FCMA':
                tqdm.write(f"--- FCMA: Re-clustering (Round {round_num+1}) | "
                          f"α={alpha_eff:.2f}, β={beta_eff:.2f}, γ={gamma_eff:.2f} ---")
                data_sim_matrix = calculate_s_data(client_dataloaders)

                # Full formula: S_cluster = α·S_model + β·S_data + γ·S_context
                combined_sim = alpha_eff * model_sim_matrix + beta_eff * data_sim_matrix

                if gamma_eff > 0:
                    context_sim_matrix = calculate_s_context(client_metadata)
                    combined_sim += gamma_eff * context_sim_matrix

                distance_matrix = 1 - combined_sim

            elif FEDERATED_MODE == 'FCAvg':
                tqdm.write(f"--- FCAvg: Re-clustering (Round {round_num+1}) ---")
                data_sim_matrix = calculate_s_data(client_dataloaders)
                combined_sim = alpha_eff * model_sim_matrix + beta_eff * data_sim_matrix
                if gamma_eff > 0:
                    context_sim_matrix = calculate_s_context(client_metadata)
                    combined_sim += gamma_eff * context_sim_matrix
                distance_matrix = 1 - combined_sim

            else:  # FedMA
                tqdm.write(f"--- FedMA: Re-clustering (Round {round_num+1}) ---")
                distance_matrix = 1 - model_sim_matrix

            num_c = NUM_CLUSTERS if FEDERATED_MODE in ['FCMA', 'FCAvg'] else 1
            clusterer = AgglomerativeClustering(
                n_clusters=num_c, metric='precomputed', linkage='average'
            )
            client_cluster_assignments = clusterer.fit_predict(distance_matrix)

            # --- Track Poisoning Isolation ---
            if poisoned_client_ids:
                iso_metrics = calculate_poisoning_isolation_rate(
                    client_cluster_assignments, poisoned_client_ids, NUM_CLIENTS
                )
                isolation_history.append({
                    'round': round_num + 1,
                    **iso_metrics
                })
                tqdm.write(
                    f"  Poisoning isolation: {iso_metrics['isolation_rate']:.2%} "
                    f"({iso_metrics['isolated_poisoned']}/{iso_metrics['total_poisoned']} isolated), "
                    f"Contamination: {iso_metrics['contamination_rate']:.2%}"
                )

            # Log cluster assignments
            tqdm.write(f"  Cluster assignments: {client_cluster_assignments.tolist()}")

        # --- Phase 1: Local Training ---
        current_local_models = []
        num_active_clients_this_round = 0
        for client_id, loader in enumerate(client_dataloaders):
            if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0 or len(loader.dataset) == 0:
                current_local_models.append(copy.deepcopy(local_models[client_id]))
                continue

            num_active_clients_this_round += 1

            if FEDERATED_MODE == 'FedAvg':
                model_to_train = copy.deepcopy(global_model)
            else:
                cluster_idx = client_cluster_assignments[client_id]
                if FEDERATED_MODE in ['FCMA', 'FCAvg'] and round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
                    prev_cluster_idx = prev_assignments[client_id]
                    if cluster_idx != prev_cluster_idx:
                        old_model_sd = local_models[client_id].state_dict()
                        new_cluster_model_sd = cluster_models[cluster_idx].state_dict()
                        blended_sd = {}
                        for key in old_model_sd:
                            blended_sd[key] = (MODEL_BLEND_WEIGHT * old_model_sd[key] +
                                             (1 - MODEL_BLEND_WEIGHT) * new_cluster_model_sd[key])
                        model_to_train = IDS_LSTM(input_dim).to(DEVICE)
                        model_to_train.load_state_dict(blended_sd)
                    else:
                        model_to_train = copy.deepcopy(cluster_models[cluster_idx])
                else:
                    model_to_train = copy.deepcopy(cluster_models[cluster_idx])

            model_to_train.lstm1.flatten_parameters()
            model_to_train.lstm2.flatten_parameters()
            current_lr = max(LEARNING_RATE * (0.95 ** (round_num // 10)), MIN_LR)
            optimizer = optim.Adam(model_to_train.parameters(), lr=current_lr)
            model_to_train.train()

            if torch.cuda.is_available() and round_num % 5 == 0:
                torch.cuda.empty_cache()

            for _ in range(LOCAL_EPOCHS):
                for data, target in loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = model_to_train(data)
                    loss = criterion(output, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()
            current_local_models.append(model_to_train)

        if len(current_local_models) == NUM_CLIENTS:
            local_models = current_local_models

        round_cost = 2 * num_active_clients_this_round * single_model_size_mb
        total_communication_cost_mb += round_cost

        # --- Phase 3: Aggregation ---
        start_time = time.time()
        active_models = [
            local_models[i] for i, l in enumerate(client_dataloaders)
            if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0
        ]

        if FEDERATED_MODE == 'FedAvg':
            if active_models:
                global_model = federated_averaging(active_models)

        elif FEDERATED_MODE == 'FedMA':
            if active_models:
                ref_model = cluster_models[0]
                agg_model = intra_cluster_fedma(active_models, ref_model, threshold=SIMILARITY_THRESHOLD)
                if agg_model:
                    cluster_models[0] = agg_model

        elif FEDERATED_MODE == 'FCAvg':
            # Clustering + simple FedAvg within clusters (ablation for RQ2)
            for cluster_id in range(NUM_CLUSTERS):
                models_in_cluster = [
                    local_models[i] for i, c_id in enumerate(client_cluster_assignments)
                    if c_id == cluster_id and i < len(local_models) and
                    hasattr(client_dataloaders[i].dataset, 'labels') and
                    len(client_dataloaders[i].dataset.labels) > 0
                ]
                if models_in_cluster:
                    agg_model = federated_averaging(models_in_cluster)
                    if agg_model:
                        cluster_models[cluster_id] = agg_model

        elif FEDERATED_MODE == 'FCMA':
            for cluster_id in range(NUM_CLUSTERS):
                models_in_cluster = [
                    local_models[i] for i, c_id in enumerate(client_cluster_assignments)
                    if c_id == cluster_id and i < len(local_models) and
                    hasattr(client_dataloaders[i].dataset, 'labels') and
                    len(client_dataloaders[i].dataset.labels) > 0
                ]
                if models_in_cluster:
                    ref_model = cluster_models[cluster_id]
                    agg_model = intra_cluster_fedma(
                        models_in_cluster, ref_model, threshold=SIMILARITY_THRESHOLD
                    )
                    if agg_model:
                        cluster_models[cluster_id] = agg_model

        aggregation_times.append(time.time() - start_time)

        # --- Evaluation ---
        if FEDERATED_MODE == 'FedAvg':
            acc, pre, rec, f1 = evaluate(global_model, global_test_loader, return_metrics=True)
        else:
            all_metrics = [evaluate(m, global_test_loader, return_metrics=True) for m in cluster_models]
            avg_metrics = np.mean(all_metrics, axis=0)
            acc, pre, rec, f1 = avg_metrics

        tqdm.write(
            f"Round {round_num+1}: Acc={acc:.4f}, F1={f1:.4f} | "
            f"Cumulative Comm. Cost: {total_communication_cost_mb:.2f} MB"
        )
        performance_history.append({
            'round': round_num + 1, 'accuracy': acc, 'f1': f1,
            'precision': pre, 'recall': rec
        })

    # ==========================================
    # --- FINAL EVALUATION AND REPORTING ---
    # ==========================================
    print("\nTraining finished. Performing final evaluations.")

    avg_agg_time = np.mean(aggregation_times) if aggregation_times else 0
    print(f"\n--- Resource Cost Analysis ({mode_label}) ---")
    print(f"Average Server-Side Aggregation Time: {avg_agg_time:.4f} seconds per round.")
    print(f"\n--- Communication Cost Summary ({mode_label}) ---")
    print(f"Total data transferred over {NUM_ROUNDS} rounds: {total_communication_cost_mb:.2f} MB")

    target_accuracy = 0.90
    rounds_to_target = -1
    cost_to_target = -1
    for record in performance_history:
        if record['accuracy'] >= target_accuracy:
            rounds_to_target = record['round']
            avg_clients_per_round = sum([
                1 for l in client_dataloaders
                if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0
            ])
            cost_to_target = 2 * rounds_to_target * avg_clients_per_round * single_model_size_mb
            break

    if rounds_to_target != -1:
        print(f"Reached {target_accuracy*100}% accuracy in {rounds_to_target} rounds.")
        print(f"Estimated communication cost to reach target: {cost_to_target:.2f} MB")
    else:
        print(f"Did not reach {target_accuracy*100}% accuracy within {NUM_ROUNDS} rounds.")

    # --- Convergence Plot ---
    print("\n--- Generating Convergence Plot ---")
    history_df = pd.DataFrame(performance_history)
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['round'], history_df['accuracy'], marker='o', linestyle='-', label='Global Accuracy')
    plt.plot(history_df['round'], history_df['f1'], marker='x', linestyle='--', label='Global F1-Score')
    plt.title(f'Convergence Plot ({mode_label}{poison_label})')
    plt.xlabel('Communication Round')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend()
    convergence_plot_path = os.path.join(results_folder, f'convergence_plot.png')
    plt.savefig(convergence_plot_path)
    plt.close()
    print(f"Convergence plot saved to {convergence_plot_path}")

    # --- Personalization Evaluation ---
    print(f"\n--- Personalization Evaluation ({mode_label}) ---")
    local_accuracies, local_f1s, local_precisions, local_recalls = [], [], [], []
    personalization_report_lines = []

    # Separate metrics for clean vs poisoned clients
    clean_accs, clean_f1s = [], []
    poisoned_accs, poisoned_f1s = [], []

    for client_id in range(NUM_CLIENTS):
        test_loader = client_test_loaders[client_id]
        if not hasattr(test_loader.dataset, 'labels') or len(test_loader.dataset) == 0:
            continue

        if FEDERATED_MODE == 'FedAvg':
            model_to_eval = global_model
            cluster_id_str = "N/A"
        else:
            cluster_id = client_cluster_assignments[client_id]
            model_to_eval = cluster_models[cluster_id]
            cluster_id_str = str(cluster_id)

        acc, pre, rec, f1 = evaluate(model_to_eval, test_loader, return_metrics=True)
        local_accuracies.append(acc)
        local_f1s.append(f1)
        local_precisions.append(pre)
        local_recalls.append(rec)

        poisoned_tag = " [POISONED]" if client_id in poisoned_client_ids else ""
        line = (f"Client {client_id} (Cluster {cluster_id_str}) -> "
                f"Acc: {acc:.4f}, F1: {f1:.4f}, Pre: {pre:.4f}, Rec: {rec:.4f}{poisoned_tag}")
        print(line)
        personalization_report_lines.append(line)

        if client_id in poisoned_client_ids:
            poisoned_accs.append(acc)
            poisoned_f1s.append(f1)
        else:
            clean_accs.append(acc)
            clean_f1s.append(f1)

    if local_accuracies:
        avg_local_acc = np.mean(local_accuracies)
        avg_local_f1 = np.mean(local_f1s)
        avg_local_pre = np.mean(local_precisions)
        avg_local_rec = np.mean(local_recalls)

        lines = [
            f"\nAverage Personalization Accuracy: {avg_local_acc:.4f}",
            f"Average Personalization F1-Score: {avg_local_f1:.4f}",
            f"Average Personalization Precision: {avg_local_pre:.4f}",
            f"Average Personalization Recall: {avg_local_rec:.4f}",
        ]

        if clean_accs:
            lines.append(f"\nClean Clients Avg Accuracy: {np.mean(clean_accs):.4f}")
            lines.append(f"Clean Clients Avg F1-Score: {np.mean(clean_f1s):.4f}")
        if poisoned_accs:
            lines.append(f"Poisoned Clients Avg Accuracy: {np.mean(poisoned_accs):.4f}")
            lines.append(f"Poisoned Clients Avg F1-Score: {np.mean(poisoned_f1s):.4f}")

        for l in lines:
            print(l)
        personalization_report_lines.extend(lines)

    # --- Confusion Matrix ---
    print("\n--- Generating Final Confusion Matrix ---")
    if FEDERATED_MODE == 'FedAvg':
        final_model = global_model
    else:
        final_model = cluster_models[0]

    y_true, y_pred = evaluate(final_model, global_test_loader, return_preds=True)

    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Final Confusion Matrix ({mode_label}{poison_label})')
        cm_plot_path = os.path.join(results_folder, f'confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Confusion matrix plot saved to {cm_plot_path}")

    # --- Poisoning Isolation Summary ---
    if isolation_history:
        print("\n--- Poisoning Isolation Over Rounds ---")
        for entry in isolation_history:
            print(f"  Round {entry['round']}: Isolation={entry['isolation_rate']:.2%}, "
                  f"Contamination={entry['contamination_rate']:.2%}")

    # --- Context Similarity Heatmap ---
    if gamma_eff > 0 and FEDERATED_MODE == 'FCMA':
        print("\n--- Generating Context Similarity Heatmap ---")
        ctx_sim = calculate_s_context(client_metadata)
        plt.figure(figsize=(8, 6))
        labels = [f"C{i}\n{client_metadata[i]['region'][:3]}\n{client_metadata[i]['vehicle_type'][:3]}"
                  for i in range(NUM_CLIENTS)]
        sns.heatmap(ctx_sim, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Contextual Similarity Matrix (S_context)')
        ctx_path = os.path.join(results_folder, 'context_similarity_heatmap.png')
        plt.savefig(ctx_path, bbox_inches='tight')
        plt.close()
        print(f"Context similarity heatmap saved to {ctx_path}")

    # --- Save Summary ---
    summary_path = os.path.join(results_folder, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"--- FEDERATED LEARNING RUN SUMMARY ---\n\n")

        f.write("--- Configuration ---\n")
        f.write(f"Federated Mode: {mode_label}\n")
        f.write(f"Number of Clients: {NUM_CLIENTS}\n")
        f.write(f"Number of Rounds: {NUM_ROUNDS}\n")
        f.write(f"Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Poisoning Rate: {poisoning_rate}\n")
        f.write(f"Use Context (S_context): {use_context}\n")

        if FEDERATED_MODE in ['FCMA', 'FCAvg', 'FedMA']:
            f.write(f"Number of Clusters: {NUM_CLUSTERS}\n")
            f.write(f"Re-clustering Interval: {RECLUSTERING_INTERVAL}\n")
            f.write(f"Similarity Threshold: {SIMILARITY_THRESHOLD}\n")
        if FEDERATED_MODE in ['FCMA', 'FCAvg']:
            f.write(f"Alpha (Model Sim): {alpha_eff:.3f}\n")
            f.write(f"Beta (Data Sim): {beta_eff:.3f}\n")
            f.write(f"Gamma (Context Sim): {gamma_eff:.3f}\n")
            f.write(f"Context Sub-Weights: w_geo={W_GEO}, w_vehicle={W_VEHICLE}, w_attack={W_ATTACK}\n")
        f.write("\n")

        f.write("--- Client Metadata ---\n")
        for meta in client_metadata:
            poisoned_tag = " [POISONED]" if meta.get('is_poisoned', False) else ""
            f.write(f"  Client {meta['client_id']}: Region={meta['region']}, "
                    f"Type={meta['vehicle_type']}, Attacks={meta['attack_exposure']}{poisoned_tag}\n")
        f.write("\n")

        f.write("--- Initial Data Distribution ---\n")
        f.write("\n".join(initial_data_distribution_report))
        f.write("\n\n")

        f.write("--- Communication Cost Analysis ---\n")
        f.write(f"Total data transferred over {NUM_ROUNDS} rounds: {total_communication_cost_mb:.2f} MB\n")
        if rounds_to_target != -1:
            f.write(f"Reached {target_accuracy*100}% accuracy in {rounds_to_target} rounds.\n")
            f.write(f"Estimated communication cost to reach target: {cost_to_target:.2f} MB\n")
        else:
            f.write(f"Did not reach {target_accuracy*100}% accuracy within {NUM_ROUNDS} rounds.\n")
        f.write("\n")

        f.write("--- Personalization Evaluation Results ---\n")
        f.write("\n".join(personalization_report_lines))
        f.write("\n\n")

        if isolation_history:
            f.write("--- Poisoning Isolation History ---\n")
            for entry in isolation_history:
                f.write(f"  Round {entry['round']}: Isolation={entry['isolation_rate']:.2%}, "
                        f"Contamination={entry['contamination_rate']:.2%}\n")
            f.write("\n")

    print(f"\nSummary report saved to {summary_path}")
    print(f"--- Experiment for {mode_label}{poison_label} Complete ---")

    return {
        'mode': mode_label,
        'poisoning_rate': poisoning_rate,
        'performance_history': performance_history,
        'isolation_history': isolation_history,
        'final_personalized_accuracy': np.mean(local_accuracies) if local_accuracies else 0,
        'final_personalized_f1': np.mean(local_f1s) if local_f1s else 0,
        'clean_client_accuracy': np.mean(clean_accs) if clean_accs else 0,
    }


# ==============================================================================
# --- EXECUTION SCRIPT ---
# ==============================================================================

def main():
    """
    Runs the full experimental suite including:
    1. Baseline comparisons (FedAvg, FedMA, FCAvg, FCMA)
    2. Ablation: FCMA with context vs. without context (γ=0)
    3. Poisoning resilience tests at different rates
    """
    all_results = []

    # --- Experiment 1: Baseline Comparison (No Poisoning) ---
    print("\n" + "="*60)
    print("  EXPERIMENT 1: BASELINE COMPARISON (No Poisoning)")
    print("="*60)
    for mode in ['FedAvg', 'FedMA', 'FCAvg', 'FCMA']:
        print(f"\n{'='*25} STARTING: {mode} {'='*25}\n")
        result = run_experiment(FEDERATED_MODE=mode, use_context=True, poisoning_rate=0.0)
        all_results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Experiment 2: Ablation - Context vs No Context ---
    print("\n" + "="*60)
    print("  EXPERIMENT 2: ABLATION - FCMA with Context vs. Without")
    print("="*60)
    for use_ctx in [True, False]:
        label = "with_context" if use_ctx else "without_context"
        print(f"\n{'='*25} ABLATION: FCMA {label} {'='*25}\n")
        result = run_experiment(FEDERATED_MODE='FCMA', use_context=use_ctx, poisoning_rate=0.0)
        all_results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Experiment 3: Poisoning Resilience ---
    print("\n" + "="*60)
    print("  EXPERIMENT 3: POISONING RESILIENCE")
    print("="*60)
    for p_rate in [0.1, 0.2, 0.3]:
        for mode in ['FedAvg', 'FCMA']:
            print(f"\n{'='*25} POISON TEST: {mode} @ {p_rate*100:.0f}% {'='*25}\n")
            result = run_experiment(FEDERATED_MODE=mode, use_context=True, poisoning_rate=p_rate)
            all_results.append(result)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Final Summary ---
    print("\n" + "="*60)
    print("  ALL EXPERIMENTS COMPLETE - SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"  {r['mode']:25s} | Poison={r['poisoning_rate']:.0%} | "
              f"Acc={r['final_personalized_accuracy']:.4f} | "
              f"F1={r['final_personalized_f1']:.4f} | "
              f"Clean Acc={r['clean_client_accuracy']:.4f}")


if __name__ == '__main__':
    main()
