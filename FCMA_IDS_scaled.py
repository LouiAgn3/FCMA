# ==============================================================================
# FCMA_IDS_scaled.py
# Scaled experiment: FedAvg vs Fed-CMA at 20 and 50 clients
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

# Will be overridden per experiment run
NUM_CLIENTS = 20
NUM_ROUNDS = 100
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SEED = 42

GRADIENT_CLIP_NORM = 1.0
MIN_LR = 0.0001

# FCMA Hyperparameters — clusters scale with clients
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

# Poisoning
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


# ==============================================================================
# --- HELPER: Compute number of clusters from number of clients ---
# ==============================================================================
def get_num_clusters(num_clients):
    """
    Scale clusters with client count. Thesis used M=5 for N=5.
    Rule of thumb: ~sqrt(N) clusters, clamped to [2, 10].
    """
    m = max(2, min(10, int(np.sqrt(num_clients))))
    return m


# ==============================================================================
# --- SYNTHETIC METADATA GENERATION ---
# ==============================================================================
REGIONS = ['Urban', 'Highway', 'Rural', 'Suburban']
VEHICLE_TYPES = ['Sedan', 'SUV', 'Truck', 'Compact']
ATTACK_TYPES_AVAILABLE = ['DoS', 'Fuzzy', 'Spoofing_Gear', 'Spoofing_RPM']


def generate_client_metadata(num_clients, seed=42):
    """
    Generates structured contextual metadata for N clients.
    Distributes clients across Urban/Highway/Rural-Suburban zones with
    correlated vehicle types and attack exposures.
    """
    rng = np.random.default_rng(seed)
    metadata = []

    # Split clients into three roughly equal groups
    n_urban = num_clients // 3
    n_highway = num_clients // 3
    # Remainder goes to rural/suburban
    n_rural = num_clients - n_urban - n_highway

    for i in range(num_clients):
        if i < n_urban:
            region = 'Urban'
            vtype = rng.choice(['Sedan', 'Compact'])
            attacks = {'DoS', 'Fuzzy'}
        elif i < n_urban + n_highway:
            region = 'Highway'
            vtype = rng.choice(['Truck', 'SUV'])
            attacks = {'Spoofing_RPM', 'Spoofing_Gear'}
        else:
            region = rng.choice(['Rural', 'Suburban'])
            vtype = rng.choice(VEHICLE_TYPES)
            n_att = rng.integers(0, len(ATTACK_TYPES_AVAILABLE) + 1)
            attacks = set(rng.choice(ATTACK_TYPES_AVAILABLE, size=n_att, replace=False)) if n_att > 0 else set()

        metadata.append({
            'client_id': i,
            'region': region,
            'vehicle_type': vtype,
            'attack_exposure': attacks,
            'is_poisoned': False,
        })

    return metadata


# ==============================================================================
# --- CONTEXTUAL SIMILARITY (Equations 2.4 – 2.7) ---
# ==============================================================================
def calculate_s_context(client_metadata, w_geo=W_GEO, w_vehicle=W_VEHICLE, w_attack=W_ATTACK):
    n = len(client_metadata)
    s_context = np.zeros((n, n))
    for i in range(n):
        s_context[i, i] = 1.0
        for j in range(i + 1, n):
            mi, mj = client_metadata[i], client_metadata[j]
            geo = 1.0 if mi['region'] == mj['region'] else 0.0
            veh = 1.0 if mi['vehicle_type'] == mj['vehicle_type'] else 0.0
            # Jaccard similarity of attack sets
            si, sj = mi['attack_exposure'], mj['attack_exposure']
            if not si and not sj:
                att = 1.0
            else:
                union = si | sj
                att = len(si & sj) / len(union) if union else 1.0
            sim = w_geo * geo + w_vehicle * veh + w_attack * att
            s_context[i, j] = s_context[j, i] = sim
    return s_context


# ==============================================================================
# --- POISONING ---
# ==============================================================================
def apply_label_flipping(client_dataloaders, client_metadata, poisoning_rate, flip_rate=1.0, seed=42):
    rng = np.random.default_rng(seed)
    num_to_poison = max(0, int(len(client_dataloaders) * poisoning_rate))
    if num_to_poison == 0:
        return set()

    eligible = [i for i, loader in enumerate(client_dataloaders)
                if hasattr(loader.dataset, 'labels') and len(loader.dataset.labels) > 0]
    poisoned_ids = set(rng.choice(eligible, size=min(num_to_poison, len(eligible)), replace=False))

    print(f"\n--- POISONING: {len(poisoned_ids)} clients: {sorted(poisoned_ids)} ---")
    for i, loader in enumerate(client_dataloaders):
        if i in poisoned_ids:
            client_metadata[i]['is_poisoned'] = True
            labels = loader.dataset.labels
            n_flip = int(len(labels) * flip_rate)
            flip_idx = rng.choice(len(labels), size=n_flip, replace=False)
            labels[flip_idx] = 1.0 - labels[flip_idx]
            print(f"  Client {i}: Flipped {n_flip}/{len(labels)} labels")
    return poisoned_ids


def calculate_poisoning_isolation_rate(assignments, poisoned_ids, num_clients):
    if not poisoned_ids:
        return {'isolation_rate': 1.0, 'contamination_rate': 0.0}
    clean_ids = set(range(num_clients)) - poisoned_ids
    isolated = 0
    for pid in poisoned_ids:
        if pid < len(assignments):
            cluster = assignments[pid]
            clean_in_cluster = any(assignments[c] == cluster for c in clean_ids if c < len(assignments))
            if not clean_in_cluster:
                isolated += 1
    contaminated = 0
    poisoned_clusters = set(assignments[p] for p in poisoned_ids if p < len(assignments))
    for c in clean_ids:
        if c < len(assignments) and assignments[c] in poisoned_clusters:
            contaminated += 1
    return {
        'isolation_rate': isolated / len(poisoned_ids),
        'contamination_rate': contaminated / len(clean_ids) if clean_ids else 0,
        'isolated': isolated, 'total_poisoned': len(poisoned_ids),
        'contaminated': contaminated, 'total_clean': len(clean_ids),
    }


# ==============================================================================
# --- DATA LOADING ---
# ==============================================================================
def load_preprocessed_data():
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        raise FileNotFoundError(f"'{PREPROCESSED_DATA_FILE}' not found.")
    print(f"Loading data from '{PREPROCESSED_DATA_FILE}'...")
    df = pd.read_csv(PREPROCESSED_DATA_FILE)
    df['Label'] = (df['Label'] != 'Normal').astype(int)
    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID']]
    return df[['Arbitration_ID', 'Label'] + feature_cols]


# ==============================================================================
# --- MODEL AND DATASET ---
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
# --- FEDERATED HELPERS ---
# ==============================================================================
def partition_data_by_can_id(df, num_clients):
    print(f"Partitioning data for {num_clients} clients by CAN ID...")
    can_ids = df['Arbitration_ID'].unique()
    np.random.shuffle(can_ids)
    client_id_map = {can_id: i % num_clients for i, can_id in enumerate(can_ids)}
    df['client_id'] = df['Arbitration_ID'].map(client_id_map)

    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID', 'client_id']]
    train_loaders, test_loaders = [], []

    for i in range(num_clients):
        client_df = df[df['client_id'] == i].drop(columns=['client_id'])
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
    return train_loaders, test_loaders


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


def intra_cluster_fedma(cluster_models, ref_model, threshold):
    if not cluster_models:
        return None
    ref_sd = copy.deepcopy(ref_model.state_dict())
    agg_sd = {}
    accum = {name: [m.state_dict()[name].clone() for m in cluster_models] for name in ref_sd}

    for name, ref_params in ref_sd.items():
        if 'weight' in name and len(ref_params.shape) > 1:
            ref_n = ref_params.view(ref_params.size(0), -1)
            sum_n = torch.zeros_like(ref_n)
            cnt = torch.zeros(ref_n.size(0), device=DEVICE)
            for other in accum[name]:
                other_n = other.view(other.size(0), -1)
                if ref_n.shape[0] != other_n.shape[0]:
                    continue
                cost = 1 - torch.nn.functional.cosine_similarity(
                    ref_n.unsqueeze(1), other_n.unsqueeze(0), dim=2)
                ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
                for r, c in zip(ri, ci):
                    if (1 - cost[r, c]) >= threshold:
                        sum_n[r] += other_n[c]
                        cnt[r] += 1
            avg_n = sum_n / cnt.clamp(min=1).unsqueeze(1)
            unmatch = torch.where(cnt == 0)[0]
            if len(unmatch) > 0:
                avg_n[unmatch] = ref_n[unmatch]
            agg_sd[name] = avg_n.view_as(ref_params)
        else:
            agg_sd[name] = torch.stack(accum[name]).mean(0)

    agg = IDS_LSTM(ref_model.lstm1.input_size).to(DEVICE)
    agg.load_state_dict(agg_sd)
    return agg


# ==============================================================================
# --- MAIN EXPERIMENT ---
# ==============================================================================
def run_experiment(federated_mode, num_clients, use_context=True, poisoning_rate=0.0):
    """
    Run a single experiment.
    federated_mode: 'FedAvg' or 'FCMA'
    num_clients: number of simulated clients (e.g. 20, 50)
    """
    num_clusters = get_num_clusters(num_clients)

    # Effective similarity weights
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

    # --- Data ---
    df = load_preprocessed_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['Label'])
    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID']]
    input_dim = len(feature_cols)

    X_test = StandardScaler().fit_transform(test_df[feature_cols].values)
    global_test_loader = DataLoader(
        CANDataset(X_test, test_df['Label'].values, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)

    client_dataloaders, client_test_loaders = partition_data_by_can_id(train_df, num_clients)

    # --- Metadata & Poisoning ---
    client_metadata = generate_client_metadata(num_clients, seed=SEED)
    poisoned_ids = apply_label_flipping(
        client_dataloaders, client_metadata, poisoning_rate, POISONING_FLIP_RATE, SEED)

    # --- Class weights ---
    label_counts = train_df['Label'].value_counts()
    pos_weight = label_counts.get(0, 1) / label_counts.get(1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))

    # --- Log data distribution ---
    dist_report = []
    print(f"\n--- Client Data Distribution ({num_clients} clients) ---")
    for cid, loader in enumerate(client_dataloaders):
        if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0:
            dist_report.append(f"Client {cid:3d}: No data")
            continue
        labels = loader.dataset.labels.numpy()
        total = len(labels)
        att = int(np.sum(labels))
        tag = " [POISONED]" if cid in poisoned_ids else ""
        line = f"Client {cid:3d}: {total:7d} samples | Attack: {att/total*100:5.1f}%{tag}"
        dist_report.append(line)
        if cid < 10 or cid in poisoned_ids:  # Print first 10 + all poisoned
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
        # Initial clustering
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

            # PCA
            active_mask = ~np.all(updates == 0, axis=1)
            n_active = np.sum(active_mask)
            if n_active > LOW_RANK_DIM:
                pca = PCA(n_components=LOW_RANK_DIM, random_state=SEED)
                pca.fit(updates[active_mask])
                M_proj = pca.components_.T
            elif updates.shape[0] > LOW_RANK_DIM:
                pca = PCA(n_components=LOW_RANK_DIM, random_state=SEED)
                pca.fit(updates)
                M_proj = pca.components_.T
            else:
                tqdm.write(f"  Round {round_num+1}: Skipping re-cluster (not enough clients for PCA)")
                M_proj = None

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

                # Track poisoning isolation
                if poisoned_ids:
                    iso = calculate_poisoning_isolation_rate(
                        client_cluster_assignments, poisoned_ids, num_clients)
                    isolation_history.append({'round': round_num + 1, **iso})
                    tqdm.write(f"    Isolation: {iso['isolation_rate']:.0%}  Contamination: {iso['contamination_rate']:.0%}")

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
                # Blending on re-cluster
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
                    agg = intra_cluster_fedma(models_in, cluster_models[cl_id], SIMILARITY_THRESHOLD)
                    if agg:
                        cluster_models[cl_id] = agg

        aggregation_times.append(time.time() - t0)

        # --- Evaluate ---
        if federated_mode == 'FedAvg':
            acc, pre, rec, f1 = evaluate(global_model, global_test_loader, return_metrics=True)
        else:
            metrics = [evaluate(m, global_test_loader, return_metrics=True) for m in cluster_models]
            acc, pre, rec, f1 = np.mean(metrics, axis=0)

        if (round_num + 1) % 10 == 0 or round_num == 0:
            tqdm.write(f"  Round {round_num+1:3d}: Acc={acc:.4f}  F1={f1:.4f}  Comm={total_comm_mb:.1f}MB")

        performance_history.append({
            'round': round_num + 1, 'accuracy': acc, 'f1': f1, 'precision': pre, 'recall': rec})

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print(f"\n--- Final Evaluation: {run_label} ---")
    print(f"Avg aggregation time: {np.mean(aggregation_times):.4f}s/round")
    print(f"Total communication: {total_comm_mb:.1f} MB")

    # Personalization
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

    # Print summary (not every client for 50)
    avg_acc = np.mean(local_accs) if local_accs else 0
    avg_f1 = np.mean(local_f1s) if local_f1s else 0
    avg_clean_acc = np.mean(clean_accs) if clean_accs else 0
    avg_clean_f1 = np.mean(clean_f1s) if clean_f1s else 0

    print(f"\n  Personalized Accuracy:  {avg_acc:.4f}")
    print(f"  Personalized F1-Score:  {avg_f1:.4f}")
    if poisoned_ids:
        print(f"  Clean Client Accuracy:  {avg_clean_acc:.4f}")
        print(f"  Clean Client F1-Score:  {avg_clean_f1:.4f}")

    # --- Convergence plot ---
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

    # --- Confusion matrix ---
    final_model = global_model if federated_mode == 'FedAvg' else cluster_models[0]
    y_true, y_pred = evaluate(final_model, global_test_loader, return_preds=True)
    if len(y_true) > 0:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix: {run_label}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'), dpi=150)
        plt.close()

    # --- Context heatmap (FCMA only) ---
    if federated_mode == 'FCMA' and g_eff > 0:
        ctx_sim = calculate_s_context(client_metadata)
        plt.figure(figsize=(max(8, num_clients * 0.4), max(6, num_clients * 0.3)))
        sns.heatmap(ctx_sim, cmap='YlOrRd', vmin=0, vmax=1)
        plt.title(f'S_context Heatmap (N={num_clients})')
        plt.savefig(os.path.join(results_folder, 'context_similarity.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # --- Summary file ---
    with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
        f.write(f"=== {run_label} ===\n\n")
        f.write(f"Clients: {num_clients}\n")
        f.write(f"Clusters: {num_clusters}\n")
        f.write(f"Rounds: {NUM_ROUNDS}\n")
        f.write(f"Mode: {federated_mode}\n")
        f.write(f"Context: {use_context}\n")
        f.write(f"Poisoning: {poisoning_rate*100:.0f}%\n")
        if federated_mode == 'FCMA':
            f.write(f"Weights: α={a_eff:.3f} β={b_eff:.3f} γ={g_eff:.3f}\n")
        f.write(f"\n--- Data Distribution ---\n")
        f.write("\n".join(dist_report))
        f.write(f"\n\n--- Communication ---\n")
        f.write(f"Total: {total_comm_mb:.1f} MB over {NUM_ROUNDS} rounds\n")
        f.write(f"\n--- Personalization ---\n")
        f.write("\n".join(pers_lines))
        f.write(f"\n\nAvg Accuracy: {avg_acc:.4f}\n")
        f.write(f"Avg F1: {avg_f1:.4f}\n")
        if poisoned_ids:
            f.write(f"Clean Accuracy: {avg_clean_acc:.4f}\n")
            f.write(f"Clean F1: {avg_clean_f1:.4f}\n")
        if isolation_history:
            f.write(f"\n--- Poisoning Isolation ---\n")
            for entry in isolation_history:
                f.write(f"Round {entry['round']}: Isolation={entry['isolation_rate']:.0%} "
                        f"Contamination={entry['contamination_rate']:.0%}\n")

    print(f"Results saved to {results_folder}/")

    return {
        'run': run_label,
        'mode': federated_mode,
        'num_clients': num_clients,
        'poisoning': poisoning_rate,
        'pers_accuracy': avg_acc,
        'pers_f1': avg_f1,
        'clean_accuracy': avg_clean_acc,
        'clean_f1': avg_clean_f1,
        'history': performance_history,
    }


# ==============================================================================
# --- EXECUTION ---
# ==============================================================================
def main():
    all_results = []

    # Run FedAvg and FCMA at 20 and 50 clients
    for n_clients in [20, 50]:
        for mode in ['FedAvg', 'FCMA']:
            print(f"\n{'#'*60}")
            print(f"#  {mode} with {n_clients} clients")
            print(f"{'#'*60}")

            result = run_experiment(
                federated_mode=mode,
                num_clients=n_clients,
                use_context=True,
                poisoning_rate=0.0,
            )
            all_results.append(result)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Comparative Summary ---
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Run':<30s} | {'Pers Acc':>8s} | {'Pers F1':>8s}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['run']:<30s} | {r['pers_accuracy']:>8.4f} | {r['pers_f1']:>8.4f}")

    # --- Comparative convergence plot ---
    plt.figure(figsize=(12, 6))
    for r in all_results:
        h = pd.DataFrame(r['history'])
        plt.plot(h['round'], h['accuracy'], label=f"{r['run']} (Acc)", linewidth=1.5)
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
