# --- SETUP AND IMPORTS ---
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

# --- CONFIGURATION ---

# Federated Learning Hyperparameters
NUM_CLIENTS = 5  # Set to 10 or 20 for full fleet simulation
NUM_ROUNDS = 50
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SEED = 42

# Stability parameters
GRADIENT_CLIP_NORM = 1.0
MIN_LR = 0.0001

# FCMA / FedMA Specific Hyperparameters
NUM_CLUSTERS = 3  # Adjusted for 5 clients (e.g., 2 Highway, 1 Urban cluster)
RECLUSTERING_INTERVAL = 10
LOW_RANK_DIM = 10
SIMILARITY_THRESHOLD = 0.1

# --- NEW: Similarity Metric Weights (The Thesis Equation) ---
# S_cluster = alpha*S_model + beta*S_data + gamma*S_context
ALPHA = 0.3  # Model Similarity
BETA = 0.4   # Data Similarity
GAMMA = 0.3  # Context Similarity

# Context Component Weights
W_GEO = 0.4
W_VEH = 0.3
W_ATT = 0.3

MODEL_BLEND_WEIGHT = 0.5 
SEQUENCE_LENGTH = 20
PREPROCESSED_DATA_FILE = 'preprocessed_paper_features_can_data.csv'

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- DATA LOADING & CONTEXT GENERATION ---

def load_preprocessed_data():
    """Loads the pre-engineered data file."""
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        raise FileNotFoundError(f"Error: '{PREPROCESSED_DATA_FILE}' not found.")

    print(f"Loading data from '{PREPROCESSED_DATA_FILE}'...")
    df = pd.read_csv(PREPROCESSED_DATA_FILE)

    # Ensure Labels are Binary (0: Normal, 1: Attack) for training
    # But keep a copy of original labels if needed for specific attack splitting
    # Here we assume the input CSV has 'Label' as 'Normal', 'DoS', 'RPM', etc.
    
    # We will process labels inside the partition function to distinguish attack types
    return df

def generate_sumo_profiles(num_clients):
    """
    Assigns a static SUMO context (Zone, Type) to each client.
    Based on Honours Project Methodology Section 2.2.2.
    """
    profiles = []
    print(f"\n--- Generating SUMO Context Profiles for {num_clients} Clients ---")
    for i in range(num_clients):
        # Deterministic assignment for reproducibility:
        # First half: Highway Trucks (High DoS/RPM threat, distinct ECU timing)
        # Second half: Urban Sedans (High Fuzzy/Gear threat, standard timing)
        if i < (num_clients // 2) + 1: 
            zone = 'Highway'
            v_type = 'Truck'
        else:
            zone = 'Urban'
            v_type = 'Sedan'
            
        profiles.append({
            'Client_ID': i,
            'Zone': zone,
            'Vehicle_Type': v_type,
            'Attack_Exposure': set() # To be filled during partitioning
        })
        print(f"Client {i}: {v_type} in {zone} Zone")
    print("----------------------------------------------------------\n")
    return profiles

def partition_data_by_context(df, profiles, sequence_length):
    """
    Partitions data based on Contextual Profiles (Label Skew & Feature Skew).
   
    """
    print("Partitioning data based on SUMO Context (Label & Feature Skew)...")
    
    # Identify Attack vs Normal rows
    # Assuming 'Label' column contains strings like 'Normal', 'DoS', 'RPM', etc.
    # If your CSV already has binary 0/1, we can't distinguish attacks easily.
    # This logic assumes the CSV has descriptive labels or we simulate the split.
    
    # Robust check: If labels are already binary 0/1, we simulate split by index
    is_binary = df['Label'].dtype != object and df['Label'].nunique() <= 2
    
    if is_binary:
        print("Note: Labels are binary. Simulating attack type split by index.")
        df_normal = df[df['Label'] == 0]
        df_attack = df[df['Label'] == 1]
    else:
        df_normal = df[df['Label'] == 'Normal']
        df_attack = df[df['Label'] != 'Normal']

    client_train_loaders = []
    client_test_loaders = []
    
    # Feature columns (exclude meta/labels)
    feature_cols = [col for col in df.columns if col not in ['Label', 'Arbitration_ID', 'Attack_Type']]

    for profile in profiles:
        client_id = profile['Client_ID']
        zone = profile['Zone']
        v_type = profile['Vehicle_Type']
        
        # --- 1. LABEL SKEW (Threat Environment) ---
        # Baseline: Everyone gets a slice of Normal data (e.g. 10%)
        client_data = df_normal.sample(frac=0.1, random_state=SEED+client_id)
        
        # Attack Allocation based on Zone
        if zone == 'Highway':
            # Highway sees "RPM/DoS" (First half of attack data)
            subset = df_attack.iloc[:len(df_attack)//2].sample(frac=0.2, replace=True)
            profile['Attack_Exposure'] = {'DoS', 'RPM'}
        else:
            # Urban sees "Fuzzy/Gear" (Second half of attack data)
            subset = df_attack.iloc[len(df_attack)//2:].sample(frac=0.2, replace=True)
            profile['Attack_Exposure'] = {'Fuzzy', 'Gear'}
            
        client_data = pd.concat([client_data, subset])
        
        # --- 2. FEATURE SKEW (ECU Heterogeneity) ---
        # Trucks have different timing. Apply scaling to 'IPT' (Inter-Packet Time).
        if v_type == 'Truck':
            # Find the IPT column (assuming it exists in features)
            ipt_cols = [c for c in feature_cols if 'IPT' in c or 'Time' in c]
            if ipt_cols:
                # Apply 15% deviation to simulate different ECU clock/bus speed
                for col in ipt_cols:
                    client_data[col] = client_data[col] * 1.15
        
        # --- Create Loaders ---
        # Convert Labels to Binary 0/1 for LSTM Training
        if not is_binary:
             y_raw = (client_data['Label'] != 'Normal').astype(int).values
        else:
             y_raw = client_data['Label'].values
             
        X_raw = client_data[feature_cols].values
        
        if len(X_raw) < sequence_length * 2:
            print(f"Warning: Client {client_id} has insufficient data.")
            client_train_loaders.append(DataLoader([], batch_size=BATCH_SIZE))
            client_test_loaders.append(DataLoader([], batch_size=BATCH_SIZE))
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=SEED
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        train_ds = CANDataset(X_train, y_train, sequence_length)
        test_ds = CANDataset(X_test, y_test, sequence_length)
        
        client_train_loaders.append(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True))
        client_test_loaders.append(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False))

    return client_train_loaders, client_test_loaders

# --- PYTORCH MODEL AND DATASET ---

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
        return self.features[idx:idx + self.sequence_length], self.labels[idx + self.sequence_length - 1].unsqueeze(-1)

# --- SIMILARITY CALCULATIONS ---

def calculate_s_context(profiles):
    """
    Calculates Contextual Similarity (S_context).
    Weighted average of S_geo, S_vehicle, and S_attack.
   
    """
    n = len(profiles)
    s_context = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            p1 = profiles[i]
            p2 = profiles[j]
            
            # S_geo: 1 if same Zone, else 0
            sim_geo = 1.0 if p1['Zone'] == p2['Zone'] else 0.0
            
            # S_vehicle: 1 if same Type, else 0
            sim_veh = 1.0 if p1['Vehicle_Type'] == p2['Vehicle_Type'] else 0.0
            
            # S_attack: Jaccard Similarity of exposure sets
            set1 = p1['Attack_Exposure']
            set2 = p2['Attack_Exposure']
            if len(set1) == 0 and len(set2) == 0:
                sim_att = 1.0 # Both empty means similar (normal traffic)
            else:
                union = len(set1.union(set2))
                sim_att = len(set1.intersection(set2)) / union if union > 0 else 0.0
            
            # Weighted Sum
            final_sim = (W_GEO * sim_geo) + (W_VEH * sim_veh) + (W_ATT * sim_att)
            
            s_context[i, j] = s_context[j, i] = final_sim
            
    return s_context

def calculate_s_data(client_dataloaders):
    """Calculates data similarity based on the ratio of Normal/Attack packets."""
    client_histograms = []
    for loader in client_dataloaders:
        if not hasattr(loader.dataset, 'labels') or len(loader.dataset.labels) == 0:
            client_histograms.append(np.array([0.5, 0.5]))
            continue
        labels = loader.dataset.labels.numpy()
        attack_count = np.sum(labels)
        total = len(labels)
        hist = np.array([1 - attack_count / total, attack_count / total])
        client_histograms.append(hist)

    s_data = np.zeros((len(client_histograms), len(client_histograms)))
    for i in range(len(client_histograms)):
        for j in range(i, len(client_histograms)):
            sim = 1 - jensenshannon(client_histograms[i], client_histograms[j])
            s_data[i, j] = s_data[j, i] = sim
    return np.nan_to_num(s_data)

def calculate_s_model(model_updates, M):
    """Calculates model similarity based on cosine similarity of low-rank projected updates."""
    if M is None or len(model_updates) == 0:
        return np.eye(len(model_updates))
    projected_updates = model_updates @ M
    norm = np.linalg.norm(projected_updates, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    cosine_sim = (projected_updates @ projected_updates.T) / (norm @ norm.T)
    return np.clip(cosine_sim, 0, 1)

# --- HELPER FUNCTIONS ---
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / 1024**2

def federated_averaging(models):
    if not models: return None
    avg_state_dict = copy.deepcopy(models[0].state_dict())
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.stack([m.state_dict()[key].float() for m in models]).mean(0)
    input_dim = models[0].lstm1.input_size
    aggregated_model = IDS_LSTM(input_dim).to(DEVICE)
    aggregated_model.load_state_dict(avg_state_dict)
    return aggregated_model

def intra_cluster_fedma(cluster_models, ref_model, threshold):
    """
    Simulated FedMA for this script. 
    (Full FedMA with Hungarian matching requires complex weight permutation logic. 
    For robustness in this integration script, we default to FedAvg within cluster, 
    but the placeholder structure allows you to swap in your full FedMA logic.)
    """
    return federated_averaging(cluster_models)

def evaluate(model, test_loader, return_metrics=False, return_preds=False):
    model.eval()
    all_preds, all_targets = [], []
    if not test_loader.dataset or len(test_loader.dataset) == 0:
        if return_preds: return np.array([]), np.array([])
        return (0.0, 0.0, 0.0, 0.0) if return_metrics else 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    if return_preds:
        return np.array(all_targets), np.array(all_preds)

    accuracy = accuracy_score(all_targets, all_preds)
    if return_metrics:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )
        return accuracy, precision, recall, f1
    return accuracy

# --- MAIN EXPERIMENT LOGIC ---
def run_experiment(FEDERATED_MODE):
    print(f"\n{'='*25} STARTING EXPERIMENT FOR: {FEDERATED_MODE} {'='*25}\n")
    
    # 1. GENERATE PROFILES & PARTITION DATA
    df = load_preprocessed_data()
    sumo_profiles = generate_sumo_profiles(NUM_CLIENTS)
    client_dataloaders, client_test_loaders = partition_data_by_context(df, sumo_profiles, SEQUENCE_LENGTH)
    
    # Prepare Global Test Set (Random sample from full DF)
    # Handle labels for global set
    is_binary = df['Label'].dtype != object and df['Label'].nunique() <= 2
    if not is_binary:
        df['Label'] = (df['Label'] != 'Normal').astype(int)
        
    df_test = df.sample(frac=0.05, random_state=SEED)
    feature_cols = [c for c in df.columns if c not in ['Label', 'Arbitration_ID', 'Attack_Type']]
    X_test_g = StandardScaler().fit_transform(df_test[feature_cols].values)
    y_test_g = df_test['Label'].values
    global_test_loader = DataLoader(CANDataset(X_test_g, y_test_g, SEQUENCE_LENGTH), batch_size=BATCH_SIZE)
    
    input_dim = len(feature_cols)

    # 2. INITIALIZE MODELS
    local_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLIENTS)]
    
    if FEDERATED_MODE == 'FedAvg':
        global_model = IDS_LSTM(input_dim).to(DEVICE)
    else:
        # FCMA/FedMA: Cluster models
        cluster_models = [IDS_LSTM(input_dim).to(DEVICE) for _ in range(NUM_CLUSTERS)]
        client_cluster_assignments = np.zeros(NUM_CLIENTS, dtype=int)
        
        # Initial Clustering
        if FEDERATED_MODE == 'FCMA':
            # FCMA uses Context + Data initially
            print("FCMA: Calculating Initial S_context and S_data...")
            s_data = calculate_s_data(client_dataloaders)
            s_context = calculate_s_context(sumo_profiles)
            
            # Initial weight: ignore model since it's random
            initial_sim = (0.5 * s_data) + (0.5 * s_context)
            clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
            client_cluster_assignments = clusterer.fit_predict(1 - initial_sim)
            print(f"Initial Cluster Assignments: {client_cluster_assignments}")
            
        elif FEDERATED_MODE == 'FedMA':
            # FedMA standard uses S_data or random start
            s_data = calculate_s_data(client_dataloaders)
            clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
            client_cluster_assignments = clusterer.fit_predict(1 - s_data)

    # 3. TRAINING LOOP
    performance_history = []
    
    for round_num in tqdm(range(NUM_ROUNDS), desc=f"Rounds ({FEDERATED_MODE})"):
        
        # --- DYNAMIC CLUSTERING (FCMA/FedMA) ---
        if FEDERATED_MODE in ['FCMA', 'FedMA'] and round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
            print(f"\n--- Dynamic Clustering Round {round_num} ---")
            
            # 1. Calculate S_model
            # Get updates relative to current cluster center
            updates = []
            for i, model in enumerate(local_models):
                center = cluster_models[client_cluster_assignments[i]]
                upd = get_flat_params(model) - get_flat_params(center)
                updates.append(upd.cpu().numpy())
            
            updates = np.array(updates)
            pca = PCA(n_components=min(LOW_RANK_DIM, len(updates)))
            if len(updates) > 1:
                pca.fit(updates)
                M = pca.components_.T
                s_model = calculate_s_model(updates, M)
            else:
                s_model = np.eye(len(updates))
                
            # 2. Calculate S_data (Can change if data streams change, here static)
            s_data = calculate_s_data(client_dataloaders)
            
            if FEDERATED_MODE == 'FCMA':
                # 3. Calculate S_context
                s_context = calculate_s_context(sumo_profiles)
                
                # 4. THE THESIS EQUATION
                # S_cluster = alpha*S_model + beta*S_data + gamma*S_context
                final_sim = (ALPHA * s_model) + (BETA * s_data) + (GAMMA * s_context)
                
            else: # FedMA just uses model similarity
                final_sim = s_model

            # 5. Cluster
            clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
            client_cluster_assignments = clusterer.fit_predict(1 - final_sim)
            print(f"New Cluster Assignments: {client_cluster_assignments}")

        # --- LOCAL TRAINING ---
        current_local_models = []
        for client_id, loader in enumerate(client_dataloaders):
            if len(loader.dataset) == 0:
                current_local_models.append(copy.deepcopy(local_models[client_id]))
                continue
                
            # Download Global/Cluster Model
            if FEDERATED_MODE == 'FedAvg':
                model_to_train = copy.deepcopy(global_model)
            else:
                cluster_idx = client_cluster_assignments[client_id]
                model_to_train = copy.deepcopy(cluster_models[cluster_idx])
            
            # Train
            model_to_train.train()
            optimizer = optim.Adam(model_to_train.parameters(), lr=LEARNING_RATE)
            criterion = nn.BCEWithLogitsLoss()
            
            for _ in range(LOCAL_EPOCHS):
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model_to_train(x), y)
                    loss.backward()
                    optimizer.step()
            
            current_local_models.append(model_to_train)
        
        local_models = current_local_models
        
        # --- AGGREGATION ---
        if FEDERATED_MODE == 'FedAvg':
            global_model = federated_averaging([m for i, m in enumerate(local_models) if len(client_dataloaders[i].dataset) > 0])
        else:
            for c_id in range(NUM_CLUSTERS):
                cluster_client_indices = [i for i, x in enumerate(client_cluster_assignments) if x == c_id]
                cluster_client_models = [local_models[i] for i in cluster_client_indices if len(client_dataloaders[i].dataset) > 0]
                
                if cluster_client_models:
                    agg_model = intra_cluster_fedma(cluster_client_models, cluster_models[c_id], SIMILARITY_THRESHOLD)
                    cluster_models[c_id] = agg_model

        # --- EVALUATION ---
        if FEDERATED_MODE == 'FedAvg':
            acc, pre, rec, f1 = evaluate(global_model, global_test_loader, return_metrics=True)
        else:
            # Evaluate all cluster models and average
            metrics = [evaluate(m, global_test_loader, return_metrics=True) for m in cluster_models]
            acc, pre, rec, f1 = np.mean(metrics, axis=0)
            
        performance_history.append({'round': round_num, 'accuracy': acc, 'f1': f1})
        if round_num % 10 == 0:
            print(f"Round {round_num}: Acc: {acc:.4f}, F1: {f1:.4f}")

    # --- SAVE RESULTS ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = f"results_{FEDERATED_MODE}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Save Convergence Plot
    df_hist = pd.DataFrame(performance_history)
    plt.figure()
    plt.plot(df_hist['round'], df_hist['accuracy'], label='Accuracy')
    plt.plot(df_hist['round'], df_hist['f1'], label='F1')
    plt.title(f"{FEDERATED_MODE} Convergence")
    plt.legend()
    plt.savefig(os.path.join(results_folder, 'convergence.png'))
    plt.close()
    
    # Save Summary
    with open(os.path.join(results_folder, 'summary.txt'), 'w') as f:
        f.write(f"Mode: {FEDERATED_MODE}\n")
        f.write(f"Final Accuracy: {acc:.4f}\n")
        f.write(f"Cluster Assignments: {client_cluster_assignments if FEDERATED_MODE != 'FedAvg' else 'N/A'}\n")

    print(f"Finished {FEDERATED_MODE}. Results saved to {results_folder}")

if __name__ == '__main__':
    # Run all modes to compare
    for mode in ['FCMA', 'FedMA', 'FedAvg']:
        run_experiment(mode)
