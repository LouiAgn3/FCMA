# --- SETUP AND IMPORTS ---
import numpy as np
import copy
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# --- CONFIGURATION PARAMETERS ---
NUM_CLIENTS = 100
NUM_ROUNDS = 100
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.005 # Keep LR the same for a fair comparison
SIMILARITY_THRESHOLD = 0.5 # Threshold is still used in the aggregation function
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- MODEL ---
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

# --- HELPER FUNCTIONS ---
# Re-using the same helper functions from the Fed-CMA script for consistency
def get_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

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

def fedma_aggregate(cluster_models, threshold=0.5): # Renamed for clarity
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
            new_layer_tensor = torch.zeros_like(ref_params)
            matches = {i: [ref_neurons[i].clone()] for i in range(num_neurons)}
            for j in range(1, len(cluster_models)):
                other_model_params = param_accumulators[name][j]
                other_neurons = other_model_params.view(other_model_params.size(0), -1)
                cost_matrix = 1 - torch.nn.functional.cosine_similarity(ref_neurons.unsqueeze(1), other_neurons.unsqueeze(0), dim=2)
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
                for r, c in zip(row_ind, col_ind):
                    similarity = 1 - cost_matrix[r, c]
                    if similarity >= threshold:
                        matches[r].append(other_neurons[c].clone())
            for i in range(num_neurons):
                avg_neuron = torch.mean(torch.stack(matches[i]), dim=0)
                new_layer_tensor.view(num_neurons, -1)[i] = avg_neuron
            aggregated_state_dict[name] = new_layer_tensor
        else:
            accumulated_params = torch.zeros_like(ref_params)
            for params in param_accumulators[name]:
                accumulated_params += params
            aggregated_state_dict[name] = accumulated_params / len(cluster_models)
    aggregated_model = SimpleCNN().to(DEVICE)
    aggregated_model.load_state_dict(aggregated_state_dict)
    return aggregated_model

# --- MAIN EXECUTION FOR BASELINE FEDMA ---
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("\nLoading and partitioning data from local files...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Use the exact same non-IID partitioning
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
    
    print("\nInitializing single global model...")
    global_model = SimpleCNN().to(DEVICE)
    accuracies = []

    print("\nStarting Federated Training (Standard FedMA)...")
    criterion = nn.CrossEntropyLoss()
    for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Rounds"):
        local_models = []
        for client_id in range(NUM_CLIENTS):
            local_model_to_train = copy.deepcopy(global_model)
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
            local_models.append(local_model_to_train)

        # Global aggregation of ALL models every round
        global_model = fedma_aggregate(local_models, threshold=SIMILARITY_THRESHOLD)

        acc = evaluate(global_model, test_loader)
        accuracies.append(acc)
        tqdm.write(f"Round {round_num}: Global Model Test Accuracy = {acc * 100:.2f}%")

    print("\nTraining finished. Saving results plot.")
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Global Model Test Accuracy')
    plt.title('Standard FedMA Convergence on MNIST (Control)')
    plt.grid(True)
    plt.savefig('baseline-fedma-convergence.png')
    plt.close()
    print("Plot saved to baseline-fedma-convergence.png")

if __name__ == '__main__':
    main()
