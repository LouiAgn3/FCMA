# --- 1. IMPORTS ---
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt

# --- 2. CONSTANTS ---
NUM_CLIENTS = 20
NUM_CLUSTERS = 4
NUM_ROUNDS = 100
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.01
BATCH_SIZE = 64
ALPHA = 0.5  # Weight for S_model
BETA = 0.5  # Weight for S_data
LOW_RANK_DIM = 50  # Low-rank approximation dimension for model updates
RECLUSTERING_INTERVAL = 10  # Re-cluster every R rounds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. DATA LOADING AND NON-IID PARTITIONING ---

# Define the transformation pipeline
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

try:
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    print("Dataset loaded successfully from local files.")
except RuntimeError:
    print("Dataset not found locally.")


# Clients get data.
client_data_indices = [[] for _ in range(NUM_CLIENTS)]
labels = train_dataset.targets
if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

# Group indices by class
indices_by_class = [np.where(labels == i)[0] for i in range(10)]

# To create a non-IID distribution where each client gets 2 classes
# we can assign classes sequentially.
for client_id in range(NUM_CLIENTS):
    # Assign two classes to each client using modulo arithmetic for cycling through classes
    class1_idx = client_id % 10
    class2_idx = (client_id + 1) % 10 # Example: Client 0 gets classes 0&1, Client 9 gets 9&0

    # Take a random half of the data for each class to avoid complete overlap
    indices1 = np.random.choice(indices_by_class[class1_idx], len(indices_by_class[class1_idx]) // 2, replace=False)
    indices2 = np.random.choice(indices_by_class[class2_idx], len(indices_by_class[class2_idx]) // 2, replace=False)

    client_data_indices[client_id].extend(indices1)
    client_data_indices[client_id].extend(indices2)


# Create data loaders for each client
client_dataloaders = [DataLoader(Subset(train_dataset, indices), batch_size=BATCH_SIZE, shuffle=True) for indices in client_data_indices]

# Define test dataset and loader (insert here, after client dataloaders)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Created {len(client_dataloaders)} non-IID client dataloaders successfully.")

# --- 4. MODEL DEFINITION ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 7*7*64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 5. FUNCTIONS ---

def calculate_s_data(dataloaders):
    """Calculate S_data using label histograms and Jensen-Shannon divergence."""
    client_histograms = []
    for dataloader in dataloaders:
        label_counts = defaultdict(int)
        for _, labels in dataloader:
            for label in labels:
                label_counts[label.item()] += 1
        histogram = np.array([label_counts[i] for i in range(10)], dtype=float)
        histogram /= histogram.sum() + 1e-10  # Normalize to probability distribution
        client_histograms.append(histogram)
    return client_histograms

def calculate_s_model(flat_updates, M):
    """Calculate S_model using low-rank projected updates and cosine similarity."""
    projected_updates = flat_updates @ M
    s_model_matrix = np.zeros((NUM_CLIENTS, NUM_CLIENTS))
    for i in range(NUM_CLIENTS):
        for j in range(i + 1, NUM_CLIENTS):
            vec_i, vec_j = projected_updates[i], projected_updates[j]
            cos_sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-10)
            s_model_matrix[i, j] = cos_sim
            s_model_matrix[j, i] = cos_sim
    return s_model_matrix

def get_flat_params(model):
    """Flatten model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def intra_cluster_fedma(models):
    """Intra-cluster aggregation using FedMA-like layer-wise matched averaging."""
    if not models:
        return None

    aggregated_model = copy.deepcopy(models[0])
    num_models = len(models)

    for layer in ['conv1', 'conv2', 'fc1', 'fc2']:  # Assuming CNN layers
        layer_weights = [getattr(m, layer).weight.data for m in models]
        # Simple average for prototype (FedMA would match neurons)
        avg_weight = sum(w for w in layer_weights) / num_models
        setattr(aggregated_model, layer).weight.data.copy_(avg_weight)

    return aggregated_model

def evaluate(model, dataloader):
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return correct / total

# --- 6. OFFLINE LOW-RANK MATRIX GENERATION ---

print("Performing offline low-rank matrix generation...")
initial_updates = []
temp_model = SimpleCNN()
for i in range(min(10, NUM_CLIENTS)): # Use 10 clients or less
    client_model = copy.deepcopy(temp_model)
    # Dummy training round
    # ... training logic here ...
    update = get_flat_params(client_model) - get_flat_params(temp_model)
    initial_updates.append(update.numpy())

pca = PCA(n_components=LOW_RANK_DIM)
pca.fit(np.array(initial_updates))
M = pca.components_.T
print("Low-rank matrix M generated.")

# --- Initialization ---
# Initialize cluster-specific models
cluster_models = [SimpleCNN().to(DEVICE) for _ in range(NUM_CLUSTERS)]
# Assign clients to clusters randomly at first
client_cluster_assignments = np.random.randint(0, NUM_CLUSTERS, NUM_CLIENTS)

# Logging
accuracies = []

# --- 10. Federated Training Rounds ---

criterion = nn.CrossEntropyLoss()
for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Rounds"):

    # --- Dynamic Clustering Phase (every R rounds) ---
    if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
        # 1. Calculate similarity matrix S
        similarity_matrix = np.zeros((NUM_CLIENTS, NUM_CLIENTS))

        # S_data part
        client_histograms = calculate_s_data(client_dataloaders)
        for i in range(NUM_CLIENTS):
            for j in range(i + 1, NUM_CLIENTS):
                js_dist = jensenshannon(client_histograms[i], client_histograms[j])
                sim = 1 - js_dist
                similarity_matrix[i, j] = BETA * sim
                similarity_matrix[j, i] = BETA * sim

        # S_model part (using updates from the last set of local models)
        # Note: We need to calculate updates relative to their cluster's global model
        model_updates = []
        for i in range(NUM_CLIENTS):
            cluster_idx = client_cluster_assignments[i]
            update = get_flat_params(local_models[i]) - get_flat_params(cluster_models[cluster_idx])
            model_updates.append(update.cpu().numpy())
            
        flat_updates = np.array(model_updates)
        s_model_matrix = calculate_s_model(flat_updates, M)
        similarity_matrix += ALPHA * s_model_matrix

        # Clustering
        distance_matrix = 1 - similarity_matrix
        clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
        client_cluster_assignments = clusterer.fit_predict(distance_matrix)
        print(f"\nRound {round_num}: Re-clustered clients. New assignments: {client_cluster_assignments}")

    # --- Local Training Phase ---
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
    
    local_models = current_local_models # Save current models for the next clustering round

    # --- Intra-Cluster Aggregation Phase ---
    for cluster_id in range(NUM_CLUSTERS):
        models_in_cluster = [local_models[i] for i, c_id in enumerate(client_cluster_assignments) if c_id == cluster_id]
        if models_in_cluster:
            aggregated_model = intra_cluster_fedma(models_in_cluster)
            cluster_models[cluster_id] = aggregated_model

    # --- Evaluation ---
    # Evaluate each cluster's model and average the accuracy
    avg_acc = np.mean([evaluate(m, test_loader) for m in cluster_models])
    accuracies.append(avg_acc)
    print(f"Round {round_num}: Average Test Accuracy = {avg_acc * 100:.2f}%")

# --- 11. Plot Results ---
plt.figure(figsize=(10, 6))
plt.plot(accuracies)
plt.xlabel('Communication Rounds')
plt.ylabel('Average Test Accuracy')
plt.title('Fed-CMA Convergence on MNIST')
plt.grid(True)
plt.show()