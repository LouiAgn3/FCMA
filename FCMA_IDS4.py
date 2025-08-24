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

print("Libraries imported successfully.")


# --- CONFIGURATION ---

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

# Similarity Metric Weights (Only for FCMA)
ALPHA = 0.3  # Increased weight for model-based similarity
BETA = 0.7   # Balanced with data-based similarity
MODEL_BLEND_WEIGHT = 0.5  # Weight for blending when switching clusters

# Data and Model Configuration
SEQUENCE_LENGTH = 20
PREPROCESSED_DATA_FILE = 'preprocessed_paper_features_can_data.csv'

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Set memory fraction to avoid OOM
    torch.cuda.set_per_process_memory_fraction(0.8)


# --- DATA LOADING ---

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


# --- PYTORCH MODEL AND DATASET ---

class IDS_LSTM(nn.Module):
    """PyTorch LSTM model for CAN Intrusion Detection, updated architecture."""
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


# --- FEDERATED LEARNING HELPERS ---

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
    """Evaluates the model's performance, can now return predictions."""
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

# --- AGGREGATION ALGORITHMS ---
def federated_averaging(models):
    """Standard Federated Averaging."""
    if not models: return None
    avg_state_dict = copy.deepcopy(models[0].state_dict())
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(0)

    input_dim = models[0].lstm1.input_size
    aggregated_model = IDS_LSTM(input_dim).to(DEVICE)
    aggregated_model.load_state_dict(avg_state_dict)
    return aggregated_model

def intra_cluster_fedma(cluster_models, ref_model, threshold):
    """Federated Matched Averaging for a cluster."""
    if not cluster_models: return None

    ref_model_state_dict = copy.deepcopy(ref_model.state_dict())
    aggregated_state_dict = {}

    param_accumulators = {name: [m.state_dict()[name].clone() for m in cluster_models] for name in ref_model_state_dict.keys()}

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


# --- MAIN EXPERIMENT LOGIC ---
def run_experiment(FEDERATED_MODE):
    """Main function to run the federated learning experiment for a given mode."""
    print(f"Running in mode: {FEDERATED_MODE}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- NEW: Create a unique results folder ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = f"results_{FEDERATED_MODE}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved in: {results_folder}")

    performance_history = []
    aggregation_times = []

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

    print("\nAccounting for skewed data by calculating class weights...")
    label_counts = train_df['Label'].value_counts()
    num_normal = label_counts.get(0, 1)
    num_attack = label_counts.get(1, 1)
    pos_weight = num_normal / num_attack
    print(f"Normal samples: {num_normal}, Attack samples: {num_attack}")
    print(f"Calculated positive weight for 'Attack' class: {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))

    if FEDERATED_MODE == 'FedAvg':
        print("Initializing global model for FedAvg...")
        global_model = IDS_LSTM(input_dim).to(DEVICE)
    elif FEDERATED_MODE == 'FedMA':
        print("Initializing single model group for FedMA (Matched Averaging)...")
        cluster_models = [IDS_LSTM(input_dim).to(DEVICE)]
        client_cluster_assignments = np.zeros(NUM_CLIENTS, dtype=int)
    elif FEDERATED_MODE == 'FCMA':
        print(f"Initializing models and clusters for {FEDERATED_MODE}...")
        cluster_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLUSTERS)]
        print("FCMA: Performing initial clustering based on data similarity...")
        data_sim_matrix = calculate_s_data(client_dataloaders)
        distance_matrix = 1 - data_sim_matrix
        clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
        client_cluster_assignments = clusterer.fit_predict(distance_matrix)
        print("Initial clustering complete.")

    local_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLIENTS)]
    
    # --- NEW: Capture initial data distribution for summary file ---
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
            line = (
                f"Client {client_id}: {total_samples} samples -> "
                f"Normal: {normal_count} ({normal_perc:.1f}%), "
                f"Attack: {attack_count} ({attack_perc:.1f}%)"
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

    # --- Training Loop ---
    for round_num in tqdm(range(NUM_ROUNDS), desc=f"Federated Rounds ({FEDERATED_MODE})"):
        if FEDERATED_MODE in ['FCMA', 'FedMA'] and round_num >= 3 and round_num % RECLUSTERING_INTERVAL == 0:
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
                tqdm.write(f"--- FCMA: Re-clustering clients based on combined similarity (Round {round_num+1}) ---")
                data_sim_matrix = calculate_s_data(client_dataloaders)
                combined_sim = ALPHA * model_sim_matrix + BETA * data_sim_matrix
                distance_matrix = 1 - combined_sim
            else: # FedMA
                tqdm.write(f"--- FedMA: Re-clustering clients based on model similarity (Round {round_num+1}) ---")
                distance_matrix = 1 - model_sim_matrix

            num_c = NUM_CLUSTERS if FEDERATED_MODE == 'FCMA' else 1
            clusterer = AgglomerativeClustering(n_clusters=num_c, metric='precomputed', linkage='average')
            client_cluster_assignments = clusterer.fit_predict(distance_matrix)

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
                if FEDERATED_MODE == 'FCMA' and round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
                    prev_cluster_idx = prev_assignments[client_id]
                    if cluster_idx != prev_cluster_idx:
                        old_model_sd = local_models[client_id].state_dict()
                        new_cluster_model_sd = cluster_models[cluster_idx].state_dict()
                        blended_sd = {}
                        for key in old_model_sd:
                            blended_sd[key] = MODEL_BLEND_WEIGHT * old_model_sd[key] + (1 - MODEL_BLEND_WEIGHT) * new_cluster_model_sd[key]
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

        start_time = time.time() 
        active_models = [local_models[i] for i, l in enumerate(client_dataloaders) if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0]

        if FEDERATED_MODE == 'FedAvg':
            if active_models:
                global_model = federated_averaging(active_models)
        elif FEDERATED_MODE == 'FedMA':
            if active_models:
                ref_model = cluster_models[0]
                agg_model = intra_cluster_fedma(active_models, ref_model, threshold=SIMILARITY_THRESHOLD)
                if agg_model:
                    cluster_models[0] = agg_model
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
                    agg_model = intra_cluster_fedma(models_in_cluster, ref_model, threshold=SIMILARITY_THRESHOLD)
                    if agg_model:
                        cluster_models[cluster_id] = agg_model

        aggregation_times.append(time.time() - start_time) 

        if FEDERATED_MODE == 'FedAvg':
            acc, pre, rec, f1 = evaluate(global_model, global_test_loader, return_metrics=True)
        else:
            all_metrics = [evaluate(m, global_test_loader, return_metrics=True) for m in cluster_models]
            avg_metrics = np.mean(all_metrics, axis=0)
            acc, pre, rec, f1 = avg_metrics
        tqdm.write(f"Round {round_num+1}: Acc={acc:.4f}, F1={f1:.4f} | Cumulative Comm. Cost: {total_communication_cost_mb:.2f} MB")
        performance_history.append({'round': round_num + 1, 'accuracy': acc, 'f1': f1, 'precision': pre, 'recall': rec})

    # --- FINAL EVALUATION AND PLOTTING ---
    print("\nTraining finished. Performing final evaluations.")

    avg_agg_time = np.mean(aggregation_times) if aggregation_times else 0
    print(f"\n--- Resource Cost Analysis ({FEDERATED_MODE}) ---")
    print(f"Average Server-Side Aggregation Time: {avg_agg_time:.4f} seconds per round.")

    print(f"\n--- Communication Cost Summary ({FEDERATED_MODE}) ---")
    print(f"Total data transferred over {NUM_ROUNDS} rounds: {total_communication_cost_mb:.2f} MB")

    target_accuracy = 0.90
    rounds_to_target = -1
    cost_to_target = -1
    for record in performance_history:
        if record['accuracy'] >= target_accuracy:
            rounds_to_target = record['round']
            avg_clients_per_round = sum([1 for l in client_dataloaders if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0])
            cost_to_target = 2 * rounds_to_target * avg_clients_per_round * single_model_size_mb
            break

    if rounds_to_target != -1:
        print(f"Reached {target_accuracy*100}% accuracy in {rounds_to_target} rounds.")
        print(f"Estimated communication cost to reach target: {cost_to_target:.2f} MB")
    else:
        print(f"Did not reach {target_accuracy*100}% accuracy within {NUM_ROUNDS} rounds.")

    print("\n--- Generating Convergence Plot ---")
    history_df = pd.DataFrame(performance_history)
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['round'], history_df['accuracy'], marker='o', linestyle='-', label='Global Accuracy')
    plt.plot(history_df['round'], history_df['f1'], marker='x', linestyle='--', label='Global F1-Score')
    plt.title(f'Convergence Plot ({FEDERATED_MODE})')
    plt.xlabel('Communication Round')
    plt.ylabel('Performance')
    plt.grid(True)
    plt.legend()
    # --- MODIFIED: Save plot to results folder ---
    convergence_plot_path = os.path.join(results_folder, f'convergence_plot_{FEDERATED_MODE}.png')
    plt.savefig(convergence_plot_path)
    plt.close() # Close the plot to free memory
    print(f"Convergence plot saved to {convergence_plot_path}")

    # --- MODIFIED: Enhanced Personalization Evaluation ---
    print(f"\n--- Personalization Evaluation ({FEDERATED_MODE}) ---")
    local_accuracies, local_f1s, local_precisions, local_recalls = [], [], [], []
    personalization_report_lines = []
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

        # Get all metrics for each client
        acc, pre, rec, f1 = evaluate(model_to_eval, test_loader, return_metrics=True)
        local_accuracies.append(acc)
        local_f1s.append(f1)
        local_precisions.append(pre)
        local_recalls.append(rec)
        
        line = f"Client {client_id} (Cluster {cluster_id_str}) -> Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}"
        print(line)
        personalization_report_lines.append(line)

    if local_accuracies:
        avg_local_acc = np.mean(local_accuracies)
        avg_local_f1 = np.mean(local_f1s)
        avg_local_pre = np.mean(local_precisions)
        avg_local_rec = np.mean(local_recalls)
        
        avg_line_1 = f"\nAverage Personalization Accuracy: {avg_local_acc:.4f}"
        avg_line_2 = f"Average Personalization F1-Score: {avg_local_f1:.4f}"
        avg_line_3 = f"Average Personalization Precision: {avg_local_pre:.4f}"
        avg_line_4 = f"Average Personalization Recall: {avg_local_rec:.4f}"
        
        print(avg_line_1)
        print(avg_line_2)
        print(avg_line_3)
        print(avg_line_4)
        
        personalization_report_lines.extend([avg_line_1, avg_line_2, avg_line_3, avg_line_4])


    print("\n--- Generating Final Confusion Matrix ---")
    if FEDERATED_MODE == 'FedAvg':
        final_model = global_model
    else:
        # Evaluate on the first cluster model as a representative for FedMA/FCMA
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
        plt.title(f'Final Confusion Matrix ({FEDERATED_MODE})')
        # --- MODIFIED: Save plot to results folder ---
        cm_plot_path = os.path.join(results_folder, f'confusion_matrix_{FEDERATED_MODE}.png')
        plt.savefig(cm_plot_path)
        plt.close() # Close the plot to free memory
        print(f"Confusion matrix plot saved to {cm_plot_path}")
    else:
        print("Test set was empty, could not generate confusion matrix.")
        
    # --- NEW: Create and save the summary.txt file ---
    summary_path = os.path.join(results_folder, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("--- FEDERATED LEARNING RUN SUMMARY ---\n\n")
        
        # 1. Configuration
        f.write("--- Configuration ---\n")
        f.write(f"Federated Mode: {FEDERATED_MODE}\n")
        f.write(f"Number of Clients: {NUM_CLIENTS}\n")
        f.write(f"Number of Rounds: {NUM_ROUNDS}\n")
        f.write(f"Local Epochs: {LOCAL_EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Seed: {SEED}\n")
        if FEDERATED_MODE in ['FCMA', 'FedMA']:
            f.write(f"Number of Clusters: {NUM_CLUSTERS}\n")
            f.write(f"Re-clustering Interval: {RECLUSTERING_INTERVAL}\n")
            f.write(f"Similarity Threshold: {SIMILARITY_THRESHOLD}\n")
        if FEDERATED_MODE == 'FCMA':
            f.write(f"Alpha (Model Sim Weight): {ALPHA}\n")
            f.write(f"Beta (Data Sim Weight): {BETA}\n")
        f.write("\n")

        # 2. Initial Data Distribution
        f.write("--- Initial Data Distribution ---\n")
        f.write("\n".join(initial_data_distribution_report))
        f.write("\n\n")
        
        # 3. Communication Cost
        f.write("--- Communication Cost Analysis ---\n")
        f.write(f"Total data transferred over {NUM_ROUNDS} rounds: {total_communication_cost_mb:.2f} MB\n")
        if rounds_to_target != -1:
            f.write(f"Reached {target_accuracy*100}% accuracy in {rounds_to_target} rounds.\n")
            f.write(f"Estimated communication cost to reach target: {cost_to_target:.2f} MB\n")
        else:
            f.write(f"Did not reach {target_accuracy*100}% accuracy within {NUM_ROUNDS} rounds.\n")
        f.write("\n")

        # 4. Personalization Results
        f.write("--- Personalization Evaluation Results ---\n")
        f.write("\n".join(personalization_report_lines))
        f.write("\n")

    print(f"\nSummary report saved to {summary_path}")
    print(f"--- Experiment for {FEDERATED_MODE} Complete ---")

# --- EXECUTION SCRIPT ---
def main():
    """Iterates through all federated modes and runs the experiment for each."""
    modes_to_run = ['FCMA', 'FedMA', 'FedAvg']
    for mode in modes_to_run:
        print(f"\n{'='*25} STARTING EXPERIMENT FOR: {mode} {'='*25}\n")
        run_experiment(FEDERATED_MODE=mode)
        print(f"\n{'='*25} FINISHED EXPERIMENT FOR: {mode} {'='*25}\n")
        
        # Clean up memory before the next run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
