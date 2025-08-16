# --- 1. SETUP AND IMPORTS ---

# Standard libraries
import numpy as np
import copy
from tqdm import tqdm # Changed from tqdm.notebook for better script compatibility

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


# --- 2. CONFIGURATION PARAMETERS ---

# Federated Learning Hyperparameters
NUM_CLIENTS = 20
NUM_ROUNDS = 100
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Fed-CMA Specific Hyperparameters
NUM_CLUSTERS = 4  # The desired number of clusters (M)
RECLUSTERING_INTERVAL = 10 # The interval 'R' for dynamic clustering
LOW_RANK_DIM = 10 # Dimensionality 'D' for the low-rank model approximation

# Similarity Metric Weights (initially, context is off)
ALPHA = 0.5 # Weight for S_model
BETA = 0.5  # Weight for S_data
GAMMA = 0.0 # Weight for S_context

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- 4. MODEL ---

class SimpleCNN(nn.Module):
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


# --- 5. HELPER FUNCTIONS FOR FED-CMA ---

# Function to get model parameters as a flat vector
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

# --- Similarity Metric Functions ---

def calculate_s_data(client_dataloaders):
    """Computes the data distribution histograms for S_data."""
    client_histograms = []
    for loader in client_dataloaders:
        labels = []
        for _, batch_labels in loader:
            labels.extend(batch_labels.tolist())
        hist = np.histogram(labels, bins=np.arange(11))[0]
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist) # Normalize
        client_histograms.append(hist)
    return np.array(client_histograms)

def calculate_s_model(model_updates, M):
    """Computes S_model using the low-rank projection matrix M."""
    # Project updates to low-rank space
    projected_updates = model_updates @ M
    # Compute cosine similarity
    norm = np.linalg.norm(projected_updates, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9 # Add a small epsilon to avoid division by zero
    cosine_sim = (projected_updates @ projected_updates.T) / (norm @ norm.T)
    return cosine_sim

# --- Evaluation Function ---

def evaluate(model, test_loader):
    """Evaluates the accuracy of a model on the test dataset."""
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

# --- Intra-Cluster Aggregation Function (FedMA) ---

def intra_cluster_fedma(cluster_models):
    """
    Performs Federated Matched Averaging on a list of models within a single cluster.
    """
    if not cluster_models:
        return None

    ref_model_state_dict = copy.deepcopy(cluster_models[0].state_dict())
    aggregated_state_dict = copy.deepcopy(ref_model_state_dict)
    
    param_accumulators = {name: [] for name in ref_model_state_dict.keys()}
    for model in cluster_models:
        for name, params in model.state_dict().items():
            param_accumulators[name].append(params.clone())

    for name, ref_params in ref_model_state_dict.items():
        if 'weight' in name and len(ref_params.shape) > 1:
            ref_neurons = ref_params.view(ref_params.size(0), -1)
            num_neurons = ref_neurons.size(0)
            new_layer_neurons = torch.zeros_like(ref_neurons)
            
            for i in range(num_neurons):
                accumulated_neuron = ref_neurons[i].clone()
                for j in range(1, len(cluster_models)):
                    other_model_params = param_accumulators[name][j]
                    other_neurons = other_model_params.view(other_model_params.size(0), -1)
                    similarities = torch.nn.functional.cosine_similarity(ref_neurons[i].unsqueeze(0), other_neurons, dim=1)
                    best_match_idx = torch.argmax(similarities).item()
                    accumulated_neuron += other_neurons[best_match_idx]
                
                new_layer_neurons[i] = accumulated_neuron / len(cluster_models)
            
            aggregated_state_dict[name] = new_layer_neurons.view(ref_params.shape)
        else:
            accumulated_params = torch.zeros_like(ref_params)
            for params in param_accumulators[name]:
                accumulated_params += params
            aggregated_state_dict[name] = accumulated_params / len(cluster_models)

    aggregated_model = SimpleCNN().to(DEVICE)
    aggregated_model.load_state_dict(aggregated_state_dict)
    return aggregated_model

# --- MAIN EXECUTION ---
def main():
    """Main function to run the Fed-CMA experiment."""
    
    # --- 3. DATA LOADING AND NON-IID PARTITIONING ---
    print("\nLoading and partitioning data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    client_data_indices = [[] for _ in range(NUM_CLIENTS)]
    labels = train_dataset.targets
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    indices_by_class = [np.where(labels == i)[0] for i in range(10)]
    for client_id in range(NUM_CLIENTS):
        class1_idx = client_id % 10
        class2_idx = (client_id + 1) % 10
        
        indices1 = np.random.choice(indices_by_class[class1_idx], len(indices_by_class[class1_idx]) // 2, replace=False)
        indices2 = np.random.choice(indices_by_class[class2_idx], len(indices_by_class[class2_idx]) // 2, replace=False)
        
        client_data_indices[client_id].extend(indices1)
        client_data_indices[client_id].extend(indices2)

    client_dataloaders = [DataLoader(Subset(train_dataset, indices), batch_size=BATCH_SIZE, shuffle=True) for indices in client_data_indices]
    print(f"Created {len(client_dataloaders)} non-IID client dataloaders successfully.")
    
    # --- Offline Step: Generate Low-Rank Matrix M ---
    print("\nPerforming offline low-rank matrix generation...")
    initial_updates = []
    temp_model = SimpleCNN()
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
    local_models = [copy.deepcopy(cluster_models[client_cluster_assignments[i]]) for i in range(NUM_CLIENTS)]
    accuracies = []

    # --- Federated Training Rounds ---
    print("\nStarting Federated Training...")
    criterion = nn.CrossEntropyLoss()
    for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Rounds"):
        if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
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
            client_cluster_assignments = clusterer.fit_predict(distance_matrix)
            print(f"\nRound {round_num}: Re-clustered clients. New assignments: {client_cluster_assignments}")

        current_local_models = []
        for client_id in range(NUM_CLIENTS):
            cluster_idx = client_cluster_assignments[client_id]
            global_model = cluster_models[cluster_idx]
            local_model = copy.deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            local_model.train()
            for epoch in range(LOCAL_EPOCHS):
                for data, target in client_dataloaders[client_id]:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            current_local_models.append(local_model)
        
        local_models = current_local_models

        for cluster_id in range(NUM_CLUSTERS):
            models_in_cluster = [local_models[i] for i, c_id in enumerate(client_cluster_assignments) if c_id == cluster_id]
            if models_in_cluster:
                aggregated_model = intra_cluster_fedma(models_in_cluster)
                cluster_models[cluster_id] = aggregated_model

        avg_acc = np.mean([evaluate(m, test_loader) for m in cluster_models])
        accuracies.append(avg_acc)
        print(f"Round {round_num}: Average Test Accuracy = {avg_acc * 100:.2f}%")

    # --- Plot and Save Results ---
    print("\nTraining finished. Saving results plot.")
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Test Accuracy')
    plt.title('Fed-CMA Convergence on MNIST')
    plt.grid(True)
    plt.savefig('fed-cma-convergence.png')
    plt.close()
    print("Plot saved to fed-cma-convergence.png")

if __name__ == '__main__':
    main()
