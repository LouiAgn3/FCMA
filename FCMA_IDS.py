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

print("Libraries imported successfully.")


# --- CONFIGURATION ---

# Select the federated learning mode: 'FedAvg', 'FedMA', or 'FCMA'
FEDERATED_MODE = 'FedMA'

# Federated Learning Hyperparameters
NUM_CLIENTS = 20
NUM_ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42

# FCMA / FedMA Specific Hyperparameters
NUM_CLUSTERS = 5
RECLUSTERING_INTERVAL = 10 # Re-cluster every 2 rounds
LOW_RANK_DIM = 10         # Dimension for PCA projection
SIMILARITY_THRESHOLD = 0.1

# Similarity Metric Weights (Only for FCMA)
ALPHA = 0.3  # Weight for model-based similarity
BETA = 0.7   # Weight for data-based similarity

# Data and Model Configuration
SEQUENCE_LENGTH = 20
PREPROCESSED_DATA_FILE = 'preprocessed_paper_features_can_data.csv'

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Running in mode: {FEDERATED_MODE}")


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
    """Partitions data among clients based on Arbitration_ID."""
    print("Partitioning data by CAN ID...")
    can_ids = df['Arbitration_ID'].unique()
    np.random.shuffle(can_ids)
    client_id_map = {can_id: i % num_clients for i, can_id in enumerate(can_ids)}

    df['client_id'] = df['Arbitration_ID'].map(client_id_map)

    client_dataloaders = []
    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID', 'client_id']]

    for i in range(num_clients):
        client_df = df[df['client_id'] == i]
        if len(client_df) < SEQUENCE_LENGTH:
            client_dataloaders.append(DataLoader([], batch_size=BATCH_SIZE))
            continue

        X = client_df[feature_cols].values
        y = client_df['Label'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset = CANDataset(X_scaled, y, SEQUENCE_LENGTH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        client_dataloaders.append(loader)

    print(f"Data partitioned into {len(client_dataloaders)} clients.")
    return client_dataloaders

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
    norm[norm == 0] = 1e-9 # Avoid division by zero
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


# --- MAIN EXECUTION ---
def main():
    """Main function to run the federated learning experiment."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- Data Loading and Partitioning ---
    df = load_preprocessed_data()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['Label'])

    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID']]
    X_test = test_df[feature_cols].values
    y_test = test_df['Label'].values
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    global_test_dataset = CANDataset(X_test_scaled, y_test, SEQUENCE_LENGTH)
    global_test_loader = DataLoader(global_test_dataset, batch_size=BATCH_SIZE)

    client_dataloaders = partition_data_by_can_id(train_df, NUM_CLIENTS)

    input_dim = len(feature_cols)

    # --- Skewed Data Handling: Calculate Class Weights ---
    print("\nAccounting for skewed data by calculating class weights...")
    label_counts = train_df['Label'].value_counts()
    num_normal = label_counts.get(0, 1) # Avoid division by zero
    num_attack = label_counts.get(1, 1)
    pos_weight = num_normal / num_attack
    print(f"Normal samples: {num_normal}, Attack samples: {num_attack}")
    print(f"Calculated positive weight for 'Attack' class: {pos_weight:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))


    # --- Model and Cluster Initialization ---
    if FEDERATED_MODE == 'FCMA':
        print(f"Initializing models and clusters for {FEDERATED_MODE}...")
        cluster_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLUSTERS)]
  
        print("FCMA: Performing initial clustering based on data similarity...")
        data_sim_matrix = calculate_s_data(client_dataloaders)
        distance_matrix = 1 - data_sim_matrix
        clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
        client_cluster_assignments = clusterer.fit_predict(distance_matrix)
        print("Initial clustering complete.")
            
    else: 
        print("Initializing global model...")
        global_model = IDS_LSTM(input_dim).to(DEVICE)

    local_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLIENTS)]

    # --- Training Loop ---
    for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Rounds"):
        
        # --- Re-clustering Logic for FCMA ---
        if FEDERATED_MODE =='FCMA' and round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
            
            # Store assignments from the start of the round to calculate updates correctly
            prev_assignments = client_cluster_assignments.copy()
            
            # 1. Calculate model updates (delta_w)
            model_updates = []
            for client_id, model in enumerate(local_models):
                # Use cluster model from the previous round as the baseline
                prev_cluster_model = cluster_models[prev_assignments[client_id]]
                update = get_flat_params(model) - get_flat_params(prev_cluster_model)
                model_updates.append(update.cpu().numpy())
            model_updates = np.array(model_updates)

            # 2. Project updates to a lower dimension
            pca = PCA(n_components=LOW_RANK_DIM, random_state=SEED)
            # Ensure we don't try to fit PCA on empty updates
            if np.any(np.all(model_updates == 0, axis=1)):
                 # Handle clients that didn't train by not including them in PCA
                 active_clients_mask = ~np.all(model_updates == 0, axis=1)
                 if np.sum(active_clients_mask) > LOW_RANK_DIM:
                      pca.fit(model_updates[active_clients_mask])
                      M = pca.components_.T
                 else: # Not enough active clients for PCA, skip re-clustering
                      M = np.random.rand(model_updates.shape[1], LOW_RANK_DIM)
            elif model_updates.shape[0] > LOW_RANK_DIM:
                pca.fit(model_updates)
                M = pca.components_.T
            else: # Not enough clients for PCA, skip re-clustering this round
                tqdm.write("--- Skipping re-clustering: Not enough clients for PCA ---")
                continue


            # 3. Calculate model similarity matrix
            model_sim_matrix = calculate_s_model(model_updates, M)

            # 4. Determine final similarity and re-cluster
            if FEDERATED_MODE == 'FCMA':
                tqdm.write(f"--- FCMA: Re-clustering clients based on combined similarity (Round {round_num+1}) ---")
                data_sim_matrix = calculate_s_data(client_dataloaders)
                combined_sim = ALPHA * model_sim_matrix + BETA * data_sim_matrix
                distance_matrix = 1 - combined_sim
            else: # FedMA
                tqdm.write(f"--- FedMA: Re-clustering clients based on model similarity (Round {round_num+1}) ---")
                distance_matrix = 1 - model_sim_matrix

            clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
            client_cluster_assignments = clusterer.fit_predict(distance_matrix)


        # --- Local Training ---
        current_local_models = []
        for client_id, loader in enumerate(client_dataloaders):
            if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0:
                current_local_models.append(copy.deepcopy(local_models[client_id]))
                continue

            if FEDERATED_MODE == 'FCMA':
                # Each client gets the model for its currently assigned cluster
                model_to_train = copy.deepcopy(cluster_models[client_cluster_assignments[client_id]])
            else: # FedAvg
                model_to_train = copy.deepcopy(global_model)

            model_to_train.lstm1.flatten_parameters()
            model_to_train.lstm2.flatten_parameters()

            optimizer = optim.Adam(model_to_train.parameters(), lr=LEARNING_RATE)
            model_to_train.train()
            for _ in range(LOCAL_EPOCHS):
                for data, target in loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = model_to_train(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            current_local_models.append(model_to_train)
        
        # Ensure list is full for next round's indexing
        if len(current_local_models) == NUM_CLIENTS:
            local_models = current_local_models

        # --- Aggregation ---
        if FEDERATED_MODE != 'FCMA':
            active_models = [local_models[i] for i, l in enumerate(client_dataloaders) if hasattr(l.dataset, 'labels') and len(l.dataset.labels) > 0]
            if active_models:
                global_model = federated_averaging(active_models)

        else:
            for cluster_id in range(NUM_CLUSTERS):
                models_in_cluster = [
                    local_models[i] for i, c_id in enumerate(client_cluster_assignments) 
                    if c_id == cluster_id and 
                    i < len(local_models) and # Safety check
                    hasattr(client_dataloaders[i].dataset, 'labels') and 
                    len(client_dataloaders[i].dataset.labels) > 0
                ]
                if models_in_cluster:
                    ref_model = cluster_models[cluster_id]
                    agg_model = intra_cluster_fedma(models_in_cluster, ref_model, threshold=SIMILARITY_THRESHOLD)
                    if agg_model:
                        cluster_models[cluster_id] = agg_model

        # --- Per-Round Evaluation ---
        if FEDERATED_MODE != 'FCMA':
            acc, pre, rec, f1 = evaluate(global_model, global_test_loader, return_metrics=True)
        else: # FedMA, FCMA
            all_metrics = [evaluate(m, global_test_loader, return_metrics=True) for m in cluster_models]
            avg_metrics = np.mean(all_metrics, axis=0)
            acc, pre, rec, f1 = avg_metrics
        tqdm.write(f"Round {round_num+1}: Acc={acc:.4f}, F1={f1:.4f}, Precision={pre:.4f}, Recall={rec:.4f}")

    # --- Final Evaluation and Confusion Matrix ---
    print("\nTraining finished. Performing final evaluation.")
    final_model = global_model if FEDERATED_MODE != 'FCMA' else cluster_models[0] # Evaluate first cluster model for FedMA/FCMA

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
        plt.savefig(f'confusion_matrix_{FEDERATED_MODE}.png')
        plt.show()
        print(f"Confusion matrix plot saved to confusion_matrix_{FEDERATED_MODE}.png")
    else:
        print("Test set was empty, could not generate confusion matrix.")


if __name__ == '__main__':
    main()
