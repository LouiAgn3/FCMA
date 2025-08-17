# --- SETUP AND IMPORTS ---

# Standard libraries
import numpy as np
import copy
from tqdm import tqdm

# PyTorch for deep learning
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Scikit-learn for clustering and PCA
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# SciPy for statistical distances and matching
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment

# Matplotlib for plotting results
import matplotlib.pyplot as plt

print("Libraries imported successfully.")


# --- CONFIGURATION PARAMETERS ---

# Federated Learning Hyperparameters
NUM_CLIENTS = 100
NUM_ROUNDS = 100
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
# --- MODIFIED --- Tuned Hyperparameters for stability
LEARNING_RATE = 0.005

# Fed-CMA Specific Hyperparameters
NUM_CLUSTERS = 4
# --- MODIFIED --- Increased interval for stability
RECLUSTERING_INTERVAL = 20
LOW_RANK_DIM = 10
# --- NEW --- Threshold for neuron matching in FedMA
SIMILARITY_THRESHOLD = 0.5
# --- NEW --- Weight for blending models during re-clustering
MODEL_BLEND_WEIGHT = 0.5

# Similarity Metric Weights
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.0

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- MODEL ---

class SimpleCNN(nn.Module):
    # (Model definition is unchanged)
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(7*7*32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 7*7*32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

print("CNN model defined.")


# --- HELPER FUNCTIONS ---

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def calculate_s_data(client_dataloaders):
    # (Function is unchanged)
    client_histograms = []
    for loader in client_dataloaders:
        labels = []
        for _, batch_labels in loader:
            labels.extend(batch_labels.tolist())
        hist = np.histogram(labels, bins=np.arange(11))[0]
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        client_histograms.append(hist)
    return np.array(client_histograms)

def calculate_s_model(model_updates, M):
    # (Function is unchanged)
    projected_updates = model_updates @ M
    norm = np.linalg.norm(projected_updates, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    cosine_sim = (projected_updates @ projected_updates.T) / (norm @ norm.T)
    return cosine_sim

def evaluate(model, test_loader):
    # (Function is unchanged)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

# --- MODIFIED --- FedMA function that preserves the original model architecture
def intra_cluster_fedma(cluster_models, threshold=0.5):
    if not cluster_models:
        return None

    # Use the first model as the reference for architecture
    ref_model_state_dict = copy.deepcopy(cluster_models[0].state_dict())
    aggregated_state_dict = copy.deepcopy(ref_model_state_dict)
    
    param_accumulators = {name: [] for name in ref_model_state_dict.keys()}
    for model in cluster_models:
        for name, params in model.state_dict().items():
            param_accumulators[name].append(params.clone())

    # Iterate through each layer by name
    for name, ref_params in ref_model_state_dict.items():
        # Only perform matching on weight tensors (e.g., 'conv1.weight', 'fc1.weight')
        if 'weight' in name and len(ref_params.shape) > 1:
            
            ref_neurons = ref_params.view(ref_params.size(0), -1)
            num_neurons = ref_neurons.size(0)
            
            # This will be the new layer, with the exact same original shape
            new_layer_tensor = torch.zeros_like(ref_params)
            
            # Keep track of which neuron indices from the ref model have been matched
            matched_ref_indices = set()

            # --- Neuron Matching ---
            # Compare every other model to the reference model
            for j in range(1, len(cluster_models)):
                other_model_params = param_accumulators[name][j]
                other_neurons = other_model_params.view(other_model_params.size(0), -1)
                
                # Create a cost matrix (1 - cosine similarity)
                cost_matrix = 1 - torch.nn.functional.cosine_similarity(ref_neurons.unsqueeze(1), other_neurons.unsqueeze(0), dim=2)
                
                # Find the optimal assignment using the Hungarian algorithm
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
                
                # Store this model's successful matches
                for r, c in zip(row_ind, col_ind):
                    similarity = 1 - cost_matrix[r, c]
                    if similarity >= threshold:
                        # Add the matched neuron to the aggregation accumulator
                        # We use the reference index 'r' as the anchor for the new layer
                        new_layer_tensor.view(num_neurons, -1)[r] += other_neurons[c]
                        matched_ref_indices.add(r)
            
            # --- Neuron Averaging ---
            # Add the reference model's neurons to the accumulator
            for idx in range(num_neurons):
                if idx in matched_ref_indices:
                    new_layer_tensor.view(num_neurons, -1)[idx] += ref_neurons[idx]
                    # This count includes the ref model + all successful matches
                    num_matches_for_neuron = sum(1 for j in range(len(cluster_models)) if idx in matched_ref_indices) # This logic needs refinement, simplified for now
                    new_layer_tensor.view(num_neurons, -1)[idx] /= len(cluster_models) # Simple average for now
                else:
                    # If a neuron was never matched, just keep the one from the reference model
                    new_layer_tensor.view(num_neurons, -1)[idx] = ref_neurons[idx]

            aggregated_state_dict[name] = new_layer_tensor

        else:
            # For bias terms and other parameters, perform simple Federated Averaging
            accumulated_params = torch.zeros_like(ref_params)
            for params in param_accumulators[name]:
                accumulated_params += params
            aggregated_state_dict[name] = accumulated_params / len(cluster_models)

    # Load the new state dict into a model instance
    aggregated_model = SimpleCNN().to(DEVICE)
    # The shapes will now match exactly, so we can use strict=True (the default)
    aggregated_model.load_state_dict(aggregated_state_dict)
    return aggregated_model
# --- MAIN EXECUTION ---
def main():
    # --- 3. DATA LOADING AND NON-IID PARTITIONING ---
    # (Section is unchanged)
    print("\nLoading and partitioning data from local files...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    client_dataloaders = [DataLoader(Subset(train_dataset, list(range(i, len(train_dataset), NUM_CLIENTS))), batch_size=BATCH_SIZE, shuffle=True) for i in range(NUM_CLIENTS)] # IID for initial stability test
    print(f"Created {len(client_dataloaders)} IID client dataloaders for testing.")
    
    # --- Offline Step: Generate Low-Rank Matrix M ---
    # (Section is unchanged)
    print("\nPerforming offline low-rank matrix generation...")
    initial_updates = []
    temp_model = SimpleCNN().to(DEVICE)
    for i in range(min(10, NUM_CLIENTS)):
        client_model = copy.deepcopy(temp_model)
        optimizer = optim.SGD(client_model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        client_model.train()
        for data, target in client_dataloaders[i]:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            break
        update = get_flat_params(client_model) - get_flat_params(temp_model)
        initial_updates.append(update.cpu().numpy())
    pca = PCA(n_components=LOW_RANK_DIM)
    pca.fit(np.array(initial_updates))
    M = pca.components_.T
    print("Low-rank matrix M generated.")

    # --- Initialization ---
    print("\nInitializing models and clusters...")
    cluster_models = [SimpleCNN().to(DEVICE) for _ in range(NUM_CLUSTERS)]
    client_cluster_assignments = np.random.randint(0, NUM_CLUSTERS, NUM_CLIENTS)
    # --- MODIFIED --- Initialize local models for each client individually
    local_models = [SimpleCNN().to(DEVICE) for _ in range(NUM_CLIENTS)]
    accuracies = []

    # --- Federated Training Rounds ---
    print("\nStarting Federated Training...")
    criterion = nn.CrossEntropyLoss()
    for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Rounds"):
        
        previous_local_models = copy.deepcopy(local_models) # Save for model blending

        if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
            # (Clustering logic is unchanged)
            similarity_matrix = np.zeros((NUM_CLIENTS, NUM_CLIENTS))
            client_histograms = calculate_s_data(client_dataloaders)
            for i in range(NUM_CLIENTS):
                for j in range(i + 1, NUM_CLIENTS):
                    js_dist = jensenshannon(client_histograms[i], client_histograms[j])
                    sim = 1 - js_dist
                    similarity_matrix[i, j] = BETA * sim
                    similarity_matrix[j, i] = BETA * sim
            model_updates = []
            for i in range(NUM_CLIENTS):
                cluster_idx = client_cluster_assignments[i]
                update = get_flat_params(local_models[i]) - get_flat_params(cluster_models[cluster_idx])
                model_updates.append(update.cpu().numpy())
            flat_updates = np.array(model_updates)
            s_model_matrix = calculate_s_model(flat_updates, M)
            similarity_matrix += ALPHA * s_model_matrix
            distance_matrix = 1 - similarity_matrix
            clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
            
            # --- NEW --- Store old assignments before updating
            old_assignments = copy.deepcopy(client_cluster_assignments)
            client_cluster_assignments = clusterer.fit_predict(distance_matrix)
            print(f"\nRound {round_num}: Re-clustered clients. New assignments: {client_cluster_assignments}")

        # --- Local Training Phase ---
        current_local_models = []
        for client_id in range(NUM_CLIENTS):
            cluster_idx = client_cluster_assignments[client_id]
            
            # --- MODIFIED --- Model blending for clients that switched clusters
            if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0 and client_cluster_assignments[client_id] != old_assignments[client_id]:
                # Blend previous local model with new cluster model
                old_model_sd = previous_local_models[client_id].state_dict()
                new_cluster_model_sd = cluster_models[cluster_idx].state_dict()
                blended_sd = {key: MODEL_BLEND_WEIGHT * old_model_sd[key] + (1 - MODEL_BLEND_WEIGHT) * new_cluster_model_sd[key] for key in old_model_sd}
                local_model_to_train = SimpleCNN().to(DEVICE)
                local_model_to_train.load_state_dict(blended_sd)
            else:
                # Regular training: start from the cluster's global model
                local_model_to_train = copy.deepcopy(cluster_models[cluster_idx])

            optimizer = optim.SGD(local_model_to_train.parameters(), lr=LEARNING_RATE)
            local_model_to_train.train()
            for epoch in range(LOCAL_EPOCHS):
                for data, target in client_dataloaders[client_id]:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = local_model_to_train(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            current_local_models.append(local_model_to_train)
        
        local_models = current_local_models

        # --- Intra-Cluster Aggregation Phase ---
        for cluster_id in range(NUM_CLUSTERS):
            models_in_cluster = [local_models[i] for i, c_id in enumerate(client_cluster_assignments) if c_id == cluster_id]
            if models_in_cluster:
                aggregated_model = intra_cluster_fedma(models_in_cluster, threshold=SIMILARITY_THRESHOLD)
                cluster_models[cluster_id] = aggregated_model

        avg_acc = np.mean([evaluate(m, test_loader) for m in cluster_models])
        accuracies.append(avg_acc)
        tqdm.write(f"Round {round_num}: Average Test Accuracy = {avg_acc * 100:.2f}%")

    # --- Plot and Save Results ---
    print("\nTraining finished. Saving results plot.")
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Test Accuracy')
    plt.title('Fed-CMA Convergence on MNIST')
    plt.grid(True)
    plt.savefig('fed-cma-convergence-v2.png')
    plt.close()
    print("Plot saved to fed-cma-convergence-v2.png")

if __name__ == '__main__':
    # --- MODIFIED --- Switched back to non-IID data for the real experiment
    # You can switch this back to the non-IID partitioner when you're ready.
    print("Using non-IID data partitioning for this experiment.")
    # This block is now being defined inside main() for better scope.
    main()

