# --- SETUP AND IMPORTS ---

# Standard libraries
import numpy as np
import copy
from tqdm import tqdm
import torch
import random

# PyTorch for deep learning
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
NUM_ROUNDS = 120
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.005
SEED = 42 # Seed for reproducibility

# Fed-CMA Specific Hyperparameters
NUM_CLUSTERS = 4
RECLUSTERING_INTERVAL = 20
LOW_RANK_DIM = 8
SIMILARITY_THRESHOLD = 0.5
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
    # Allow dynamic input size for fc1
    def __init__(self, fc1_in_features=7*7*32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Store it
        self.fc1_in_features = fc1_in_features
        self.fc1 = nn.Linear(fc1_in_features, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # Use the stored feature size
        x = x.view(-1, self.fc1_in_features)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

print("CNN model defined.")


# --- HELPER FUNCTIONS ---

def partition_data_non_iid_dirichlet(dataset, num_clients, alpha=0.5):
    print(f"Partitioning data into {num_clients} non-IID clients with alpha={alpha}...")
    try:
        labels = dataset.targets.numpy()
    except AttributeError:
        labels = np.array([sample[1] for sample in dataset])

    num_classes = len(np.unique(labels))
    idx_by_class = [np.where(labels == i)[0] for i in range(num_classes)]
    class_proportions = np.random.dirichlet([alpha] * num_classes, num_clients)
    client_idx_map = {i: [] for i in range(num_clients)}
    class_start_idx = [0] * num_classes

    for client_i in range(num_clients):
        for class_j in range(num_classes):
            proportion = class_proportions[client_i, class_j]
            num_to_assign = int(proportion * len(idx_by_class[class_j]))
            start = class_start_idx[class_j]
            end = start + num_to_assign
            assigned_indices = idx_by_class[class_j][start:end]
            client_idx_map[client_i].extend(assigned_indices)
            class_start_idx[class_j] = end
        np.random.shuffle(client_idx_map[client_i])

    client_dataloaders = []
    for i in range(num_clients):
        indices = client_idx_map[i]
        if len(indices) > 0:
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
            client_dataloaders.append(loader)

    print(f"Data partitioning complete. Created {len(client_dataloaders)} non-empty client dataloaders.")
    return client_dataloaders

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def calculate_s_data(client_dataloaders):
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
    projected_updates = model_updates @ M
    norm = np.linalg.norm(projected_updates, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    cosine_sim = (projected_updates @ projected_updates.T) / (norm @ norm.T)
    return cosine_sim

def evaluate(model, test_loader):
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

def visualize_client_data_distribution(dataloaders, num_clients_to_show=5):
    num_clients_to_show = min(num_clients_to_show, len(dataloaders))
    if num_clients_to_show == 0:
        print("No clients to visualize.")
        return
    fig, axes = plt.subplots(1, num_clients_to_show, figsize=(20, 4), sharey=True)
    fig.suptitle('Client Data Distributions (Non-IID)')
    for i in range(num_clients_to_show):
        loader = dataloaders[i]
        labels = []
        for _, batch_labels in loader:
            labels.extend(batch_labels.tolist())
        hist = np.histogram(labels, bins=np.arange(11))[0]
        axes[i].bar(np.arange(10), hist)
        axes[i].set_title(f"Client {i}")
        axes[i].set_xlabel("Class (Digit)")
        if i == 0:
            axes[i].set_ylabel("Number of Samples")
    plt.savefig('client_data_distribution.png')
    plt.close()
    print("Saved client data distribution plot to 'client_data_distribution.png'")

def intra_cluster_fedma(cluster_models, threshold=0.5):
    if not cluster_models:
        return None

    ref_model_state_dict = copy.deepcopy(cluster_models[0].state_dict())
    aggregated_state_dict = {}
    
    param_accumulators = {name: [] for name in ref_model_state_dict.keys()}
    for model in cluster_models:
        for name, params in model.state_dict().items():
            param_accumulators[name].append(params.clone())


    for name, ref_params in ref_model_state_dict.items():
        if 'weight' in name and len(ref_params.shape) > 1:
            all_neurons_in_layer = [p.view(p.size(0), -1) for p in param_accumulators[name]]
            ref_neurons = all_neurons_in_layer[0]
            num_ref_neurons = ref_neurons.size(0)

            matches = {i: [ref_neurons[i]] for i in range(num_ref_neurons)}
            unmatched_neurons = []
            
            client_matched_indices = [set() for _ in range(len(all_neurons_in_layer))]
            client_matched_indices[0] = set(range(num_ref_neurons))


            for client_idx in range(1, len(all_neurons_in_layer)):
                other_neurons = all_neurons_in_layer[client_idx]
                cost_matrix = 1 - torch.nn.functional.cosine_similarity(ref_neurons.unsqueeze(1), other_neurons.unsqueeze(0), dim=2)
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

                for r, c in zip(row_ind, col_ind):
                    if 1 - cost_matrix[r, c].item() >= threshold:
                        matches[r].append(other_neurons[c])
                        client_matched_indices[client_idx].add(c)

            for client_idx in range(len(all_neurons_in_layer)):
                other_neurons = all_neurons_in_layer[client_idx]
                for c in range(other_neurons.size(0)):
                    if c not in client_matched_indices[client_idx]:
                        unmatched_neurons.append(other_neurons[c])
            
            aggregated_neurons_list = []
            if matches:
                aggregated_neurons_list = [torch.mean(torch.stack(match_group), dim=0) for match_group in matches.values() if match_group]

            if aggregated_neurons_list:
                aggregated_neurons_tensor = torch.stack(aggregated_neurons_list)
                for u_neuron in unmatched_neurons:
                    similarities = torch.nn.functional.cosine_similarity(u_neuron.unsqueeze(0), aggregated_neurons_tensor)
                    max_sim, best_match_idx = torch.max(similarities, dim=0)
                    if max_sim > threshold:
                        match_key = list(matches.keys())[best_match_idx.item()]
                        original_group_size = len(matches.get(match_key, []))
                        if original_group_size > 0:
                            old_sum = aggregated_neurons_tensor[best_match_idx] * original_group_size
                            new_avg = (old_sum + u_neuron) / (original_group_size + 1)
                            aggregated_neurons_tensor[best_match_idx] = new_avg
                            matches[match_key].append(u_neuron)

                    else:
                        aggregated_neurons_list.append(u_neuron)
                        aggregated_neurons_tensor = torch.stack(aggregated_neurons_list)
            else:
                 aggregated_neurons_list.extend(unmatched_neurons)

            if aggregated_neurons_list:
                final_layer_tensor = torch.stack(aggregated_neurons_list)
                original_shape = list(ref_params.shape)
                original_shape[0] = final_layer_tensor.shape[0]
                aggregated_state_dict[name] = final_layer_tensor.view(original_shape)
            else:
                aggregated_state_dict[name] = ref_params
        else:
            accumulated_params = torch.stack(param_accumulators[name])
            aggregated_state_dict[name] = torch.mean(accumulated_params, dim=0)

    new_fc1_in_features = 7 * 7 * aggregated_state_dict['conv2.weight'].shape[0]
    aggregated_model = SimpleCNN(fc1_in_features=new_fc1_in_features).to(DEVICE)

    if 'fc1.weight' in aggregated_state_dict and aggregated_state_dict['conv2.weight'].shape[0] != ref_model_state_dict['conv2.weight'].shape[0]:
        old_fc1_weight = aggregated_state_dict['fc1.weight']
        new_fc1_weight = torch.randn(old_fc1_weight.shape[0], new_fc1_in_features).to(DEVICE)
        min_in_features = min(old_fc1_weight.shape[1], new_fc1_in_features)
        new_fc1_weight[:, :min_in_features] = old_fc1_weight[:, :min_in_features]
        aggregated_state_dict['fc1.weight'] = new_fc1_weight

    if 'fc1.weight' in aggregated_state_dict and 'fc2.weight' in aggregated_state_dict and aggregated_state_dict['fc1.weight'].shape[0] != ref_model_state_dict['fc1.weight'].shape[0]:
        new_fc1_out_features = aggregated_state_dict['fc1.weight'].shape[0]
        aggregated_state_dict['fc1.bias'] = torch.zeros(new_fc1_out_features).to(DEVICE)
        old_fc2_weight = aggregated_state_dict['fc2.weight']
        new_fc2_weight = torch.randn(old_fc2_weight.shape[0], new_fc1_out_features).to(DEVICE)
        min_in_features = min(old_fc2_weight.shape[1], new_fc1_out_features)
        new_fc2_weight[:, :min_in_features] = old_fc2_weight[:, :min_in_features]
        aggregated_state_dict['fc2.weight'] = new_fc2_weight

    try:
        aggregated_model.load_state_dict(aggregated_state_dict)
    except RuntimeError as e:
        print("--- ERROR LOADING STATE DICT ---")
        print(f"Error: {e}")
        return cluster_models[0]
        
    return aggregated_model


# --- MAIN EXECUTION ---
def main():
    # --- Seeding for Reproducibility ---
    print(f"Using random seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        # The following two lines are essential for ensuring deterministic behavior on a GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Using non-IID data partitioning for this experiment.")

    print("\nLoading and partitioning data from local files...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    client_dataloaders = partition_data_non_iid_dirichlet(train_dataset, NUM_CLIENTS, alpha=0.5)
    visualize_client_data_distribution(client_dataloaders)
    
    actual_num_clients = len(client_dataloaders)
    if actual_num_clients == 0:
        print("No clients have data after partitioning. Exiting.")
        return
    print(f"Proceeding with {actual_num_clients} active clients.")
    
    print("\nPerforming offline low-rank matrix generation...")
    initial_updates = []
    temp_model = SimpleCNN().to(DEVICE)
    
    # <-- LOGIC CHANGE HERE: Ensure enough samples are collected for PCA
    num_pca_samples = min(LOW_RANK_DIM, actual_num_clients) 
    
    for i in range(num_pca_samples):
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
    
    if not initial_updates or len(initial_updates) < LOW_RANK_DIM:
        print(f"Not enough client samples ({len(initial_updates)}) to perform PCA with {LOW_RANK_DIM} dimensions. Exiting.")
        return
        
    pca = PCA(n_components=LOW_RANK_DIM)
    pca.fit(np.array(initial_updates))
    M = pca.components_.T
    print("Low-rank matrix M generated.")

    print("\nInitializing models and clusters...")
    cluster_models = [SimpleCNN().to(DEVICE) for _ in range(NUM_CLUSTERS)]
    client_cluster_assignments = np.random.randint(0, NUM_CLUSTERS, actual_num_clients)
    local_models = [SimpleCNN().to(DEVICE) for _ in range(actual_num_clients)]
    accuracies = []

    print("\nStarting Federated Training...")
    criterion = nn.CrossEntropyLoss()
    for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Rounds"):
        previous_local_models = copy.deepcopy(local_models)

        if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
            similarity_matrix = np.zeros((actual_num_clients, actual_num_clients))
            client_histograms = calculate_s_data(client_dataloaders)
            for i in range(actual_num_clients):
                for j in range(i + 1, actual_num_clients):
                    js_dist = jensenshannon(client_histograms[i], client_histograms[j])
                    sim = 1 - js_dist
                    similarity_matrix[i, j] = BETA * sim
                    similarity_matrix[j, i] = BETA * sim
            
            model_updates = []
            for i in range(actual_num_clients):
                cluster_idx = client_cluster_assignments[i]
                update = get_flat_params(local_models[i]) - get_flat_params(cluster_models[cluster_idx])
                model_updates.append(update.cpu().numpy())
            
            flat_updates = np.array(model_updates)
            s_model_matrix = calculate_s_model(flat_updates, M)
            similarity_matrix += ALPHA * s_model_matrix
            distance_matrix = 1 - similarity_matrix
            clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
            
            old_assignments = copy.deepcopy(client_cluster_assignments)
            client_cluster_assignments = clusterer.fit_predict(distance_matrix)
            print(f"\nRound {round_num}: Re-clustered clients.")

        current_local_models = []
        for client_id in range(actual_num_clients):
            cluster_idx = client_cluster_assignments[client_id]
            
            local_model_to_train = SimpleCNN().to(DEVICE)
            local_model_to_train.load_state_dict(cluster_models[cluster_idx].state_dict())

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

        for cluster_id in range(NUM_CLUSTERS):
            models_in_cluster = [local_models[i] for i, c_id in enumerate(client_cluster_assignments) if c_id == cluster_id]
            if models_in_cluster:
                aggregated_model = intra_cluster_fedma(models_in_cluster, threshold=SIMILARITY_THRESHOLD)
                if aggregated_model:
                    cluster_models[cluster_id] = aggregated_model

        avg_acc = np.mean([evaluate(m, test_loader) for m in cluster_models])
        accuracies.append(avg_acc)
        tqdm.write(f"Round {round_num}: Average Test Accuracy = {avg_acc * 100:.2f}%")

    print("\nTraining finished. Saving results plot.")
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Average Test Accuracy')
    plt.title('Fed-CMA Convergence on MNIST (Non-IID)')
    plt.grid(True)
    plt.savefig('fed-cma-convergence-non-iid.png')
    plt.close()
    print("Plot saved to fed-cma-convergence-non-iid.png")

if __name__ == '__main__':
    main()
