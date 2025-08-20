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
from torch.utils.data import DataLoader, Subset, random_split
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
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SEED = 42 # Seed for reproducibility

# Fed-CMA Specific Hyperparameters
NUM_CLUSTERS = 10
RECLUSTERING_INTERVAL = 5
LOW_RANK_DIM = 10
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
    def __init__(self, fc1_in_features=8*8*32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1_in_features = fc1_in_features
        self.fc1 = nn.Linear(fc1_in_features, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.fc1_in_features)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

print("CNN model defined for CIFAR-10.")


# --- HELPER FUNCTIONS ---

def partition_data_non_iid_dirichlet(dataset, num_clients, alpha=0.1, train_split_ratio=0.8):
    print(f"Partitioning data into {num_clients} non-IID clients with alpha={alpha}...")
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        labels = np.array([y for _, y in dataset])
    num_classes = int(labels.max()) + 1

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        props = np.random.dirichlet([alpha] * num_clients)
        counts = np.floor(props * len(idx_c)).astype(int)
        diff = len(idx_c) - counts.sum()
        if diff > 0:
            frac = props * len(idx_c) - counts
            counts[np.argsort(-frac)[:diff]] += 1
        start = 0
        for i, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[i].extend(idx_c[start:start+cnt].tolist())
                start += cnt

    client_trainloaders, client_testloaders = [], []
    for inds in client_indices:
        if inds:
            np.random.shuffle(inds)
            split_point = int(len(inds) * train_split_ratio)
            train_inds = inds[:split_point]
            test_inds = inds[split_point:]

            if train_inds:
                client_trainloaders.append(DataLoader(Subset(dataset, train_inds), batch_size=BATCH_SIZE, shuffle=True))
            else:
                client_trainloaders.append(DataLoader([], batch_size=BATCH_SIZE))

            if test_inds:
                client_testloaders.append(DataLoader(Subset(dataset, test_inds), batch_size=BATCH_SIZE, shuffle=False))
            else:
                client_testloaders.append(DataLoader([], batch_size=BATCH_SIZE))

    print(f"Data partitioning complete. Created {len(client_trainloaders)} train/test loader pairs.")
    return client_trainloaders, client_testloaders

def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def calculate_s_data(client_dataloaders):
    client_histograms = []
    for loader in client_dataloaders:
        if not loader.dataset: continue
        labels = []
        if isinstance(loader.dataset, Subset):
            targets = np.array(loader.dataset.dataset.targets)
            sub_labels = targets[loader.dataset.indices]
            labels.extend(sub_labels.tolist())
        else:
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
    
    # Normalize the similarity matrix to be in the same range [0, 1]
    normalized_sim = np.clip(cosine_sim, 0, 1)
    return normalized_sim
    
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    if not test_loader.dataset: return 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total if total > 0 else 0.0

def visualize_client_data_distribution(dataloaders, num_clients_to_show=5):
    num_clients_to_show = min(num_clients_to_show, len(dataloaders))
    if num_clients_to_show == 0:
        print("No clients to visualize.")
        return
    fig, axes = plt.subplots(1, num_clients_to_show, figsize=(20, 4), sharey=True)
    fig.suptitle('Client Data Distributions (Non-IID)')
    for i in range(num_clients_to_show):
        loader = dataloaders[i]
        if not loader.dataset: continue
        labels = []
        if isinstance(loader.dataset, Subset):
            targets = np.array(loader.dataset.dataset.targets)
            sub_labels = targets[loader.dataset.indices]
            labels.extend(sub_labels.tolist())
        else:
            for _, batch_labels in loader:
                labels.extend(batch_labels.tolist())
        hist = np.histogram(labels, bins=np.arange(11))[0]
        axes[i].bar(np.arange(10), hist)
        axes[i].set_title(f"Client {i}")
        axes[i].set_xlabel("Class")
        if i == 0:
            axes[i].set_ylabel("Number of Samples")
    plt.savefig('client_data_distribution_cifar10.png')
    plt.close()
    print("Saved client data distribution plot to 'client_data_distribution_cifar10.png'")

def intra_cluster_fedma(cluster_models, ref_model, threshold=0.5):
    if not cluster_models:
        return None

    ref_model_state_dict = copy.deepcopy(ref_model.state_dict())
    aggregated_state_dict = {}

    param_accumulators = {name: [] for name in ref_model_state_dict.keys()}
    for model in cluster_models:
        for name, params in model.state_dict().items():
            param_accumulators[name].append(params.clone())

    for name, ref_params in ref_model_state_dict.items():
        if 'weight' in name and len(ref_params.shape) > 1:
            ref_neurons = ref_params.view(ref_params.size(0), -1)
            num_neurons = ref_neurons.size(0)
            
            sum_neurons = torch.zeros_like(ref_neurons)
            count_neurons = torch.zeros(num_neurons, device=ref_neurons.device)

            for j in range(len(cluster_models)):
                other_params = param_accumulators[name][j]
                other_neurons = other_params.view(other_params.size(0), -1)

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
            accumulated_params = torch.stack(param_accumulators[name])
            aggregated_state_dict[name] = torch.mean(accumulated_params, dim=0)

    INPUT_SHAPE = (3, 32, 32)
    with torch.no_grad():
        temp = SimpleCNN(fc1_in_features=1).to(DEVICE)
        sd = temp.state_dict()
        sd['conv1.weight'] = aggregated_state_dict['conv1.weight']
        sd['conv1.bias']   = aggregated_state_dict['conv1.bias']
        sd['conv2.weight'] = aggregated_state_dict['conv2.weight']
        sd['conv2.bias']   = aggregated_state_dict['conv2.bias']
        temp.load_state_dict(sd, strict=False)
        dummy = torch.zeros(1, *INPUT_SHAPE).to(DEVICE)
        feat = temp.pool2(temp.relu2(temp.conv2(temp.pool1(temp.relu1(temp.conv1(dummy))))))
        new_fc1_in_features = int(feat.numel())

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
        return ref_model

    return aggregated_model


# --- MAIN EXECUTION ---
def main():
    print(f"Using random seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Using non-IID data partitioning for this experiment.")

    print("\nLoading and partitioning CIFAR-10 data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    global_test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    client_trainloaders, client_testloaders = partition_data_non_iid_dirichlet(train_dataset, NUM_CLIENTS, alpha=0.1)
    visualize_client_data_distribution(client_trainloaders)

    actual_num_clients = len([d for d in client_trainloaders if d.dataset])
    if actual_num_clients == 0:
        print("No clients have data after partitioning. Exiting.")
        return
    print(f"Proceeding with {actual_num_clients} active clients.")

    print("\nPerforming offline low-rank matrix generation (for initialization)...")
    initial_updates = []
    temp_model = SimpleCNN().to(DEVICE)

    # Use a subset of clients for initial PCA
    clients_for_pca = [i for i, d in enumerate(client_trainloaders) if d.dataset]
    num_pca_samples = min(LOW_RANK_DIM * 2, len(clients_for_pca)) # Use more samples for stability
    
    for i in random.sample(clients_for_pca, num_pca_samples):
        client_model = copy.deepcopy(temp_model)
        optimizer = optim.SGD(client_model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        client_model.train()
        # Train for one step to get a meaningful update
        try:
            data, target = next(iter(client_trainloaders[i]))
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            update = get_flat_params(client_model) - get_flat_params(temp_model)
            initial_updates.append(update.cpu().numpy())
        except StopIteration:
            continue # Skip if client has no data after all

    if not initial_updates or len(initial_updates) < LOW_RANK_DIM:
        print("Not enough client samples for initial PCA. Exiting.")
        return
    
    n_comp = LOW_RANK_DIM
    pca = PCA(n_components=n_comp)
    pca.fit(np.array(initial_updates))
    M = pca.components_.T
    print("Initial low-rank matrix M generated.")

    print("\nInitializing models and clusters...")
    print("Performing initial data-based clustering...")
    client_histograms = calculate_s_data(client_trainloaders)
    initial_data_sim_matrix = np.zeros((len(client_trainloaders), len(client_trainloaders)))
    for i in range(len(client_trainloaders)):
        for j in range(i + 1, len(client_trainloaders)):
            if i < len(client_histograms) and j < len(client_histograms):
                js_dist = jensenshannon(client_histograms[i], client_histograms[j])
                sim = 1 - js_dist
                initial_data_sim_matrix[i, j] = sim
                initial_data_sim_matrix[j, i] = sim
            
    distance_matrix = 1 - initial_data_sim_matrix
    clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
    client_cluster_assignments = clusterer.fit_predict(distance_matrix)
    print("Initial clustering complete.")
    
    cluster_models = [SimpleCNN().to(DEVICE) for _ in range(NUM_CLUSTERS)]
    local_models = [SimpleCNN().to(DEVICE) for _ in range(len(client_trainloaders))]
    
    cluster_accuracies = []
    personalized_accuracies = []

    print("\nStarting Federated Training...")
    criterion = nn.CrossEntropyLoss()
    for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Rounds"):
        if round_num > 0 and round_num % RECLUSTERING_INTERVAL == 0:
            tqdm.write(f"\nRound {round_num}: Re-clustering clients and updating PCA matrix...")

            model_updates = []
            active_client_indices = []
            for i in range(len(client_trainloaders)):
                 if client_trainloaders[i].dataset:
                    cluster_idx = client_cluster_assignments[i]
                    update = get_flat_params(local_models[i]) - get_flat_params(cluster_models[cluster_idx])
                    model_updates.append(update.cpu().numpy())
                    active_client_indices.append(i)
            
            flat_updates = np.array(model_updates)

            # --- DYNAMIC PCA CALCULATION ---
            if flat_updates.shape[0] >= LOW_RANK_DIM:
                n_comp = LOW_RANK_DIM
                pca = PCA(n_components=n_comp)
                pca.fit(flat_updates)
                M = pca.components_.T
                tqdm.write("Dynamically updated PCA matrix M.")
            else:
                tqdm.write("Not enough client updates to re-calculate PCA, using previous M.")

            s_model_matrix_full = calculate_s_model(flat_updates, M)
            
            s_data_matrix = np.zeros((len(client_trainloaders), len(client_trainloaders)))
            client_histograms = calculate_s_data(client_trainloaders)
            for i in range(len(client_histograms)):
                for j in range(i + 1, len(client_histograms)):
                    js_dist = jensenshannon(client_histograms[i], client_histograms[j])
                    sim = 1 - js_dist
                    s_data_matrix[i,j] = sim
                    s_data_matrix[j,i] = sim
            
            # Combine similarity matrices
            similarity_matrix = np.zeros_like(s_data_matrix)
            # Map s_model back to the full client matrix
            for i_idx, i_original in enumerate(active_client_indices):
                for j_idx, j_original in enumerate(active_client_indices):
                    similarity_matrix[i_original, j_original] = s_model_matrix_full[i_idx, j_idx]
            
            similarity_matrix = ALPHA * similarity_matrix + BETA * s_data_matrix
            
            distance_matrix = 1 - similarity_matrix
            clusterer = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, metric='precomputed', linkage='average')
            client_cluster_assignments = clusterer.fit_predict(distance_matrix)
            tqdm.write(f"Re-clustering complete.")

        current_local_models = []
        for client_id in range(len(client_trainloaders)):
            if not client_trainloaders[client_id].dataset:
                current_local_models.append(copy.deepcopy(local_models[client_id]))
                continue
                
            cluster_idx = client_cluster_assignments[client_id]
            local_model_to_train = copy.deepcopy(cluster_models[cluster_idx])
            optimizer = optim.SGD(local_model_to_train.parameters(), lr=LEARNING_RATE)
            local_model_to_train.train()
            for epoch in range(LOCAL_EPOCHS):
                for data, target in client_trainloaders[client_id]:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = local_model_to_train(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            current_local_models.append(local_model_to_train)
        local_models = current_local_models

        round_personalized_accs = []
        for client_id in range(len(client_trainloaders)):
            acc = evaluate(local_models[client_id], client_testloaders[client_id])
            round_personalized_accs.append(acc)
        avg_personalized_acc = np.mean(round_personalized_accs)
        personalized_accuracies.append(avg_personalized_acc)

        for cluster_id in range(NUM_CLUSTERS):
            models_in_cluster = [local_models[i] for i, c_id in enumerate(client_cluster_assignments) if c_id == cluster_id and client_trainloaders[i].dataset]
            
            if models_in_cluster:
                reference_model = cluster_models[cluster_id]
                aggregated_model = intra_cluster_fedma(models_in_cluster, reference_model, threshold=SIMILARITY_THRESHOLD)
                if aggregated_model:
                    cluster_models[cluster_id] = aggregated_model

        avg_cluster_acc = np.mean([evaluate(m, global_test_loader) for m in cluster_models])
        cluster_accuracies.append(avg_cluster_acc)
        
        tqdm.write(f"Round {round_num}: Avg Cluster Acc = {avg_cluster_acc * 100:.2f}% | "
                   f"Avg Personalized Acc = {avg_personalized_acc * 100:.2f}%")


    print("\nTraining finished. Saving results plot.")
    plt.figure(figsize=(12, 7))
    plt.plot(cluster_accuracies, label='Average Cluster Accuracy (on Global Test Set)')
    plt.plot(personalized_accuracies, label='Average Personalized Accuracy (on Local Test Sets)', linestyle='--')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title('Fed-CMA: Cluster vs. Personalized Accuracy on CIFAR-10 (Non-IID)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fed-cma-accuracies-cifar10-non-iid_dynamic_pca.png')
    plt.close()
    print("Plot saved to fed-cma-accuracies-cifar10-non-iid_dynamic_pca.png")

if __name__ == '__main__':
    main()
