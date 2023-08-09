import pickle
import os
from torch.nn.functional import mse_loss
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, explained_variance_score
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import to_dense_adj
from egnn_pytorch import EGNN_Network

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'datasetQM9.pkl')

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

BATCH_SIZE = 64
dataset = []
for atom_features, atom_positions, adjacency_matrix, bond_features, _, quadrupole in data:
    edge_index = torch.tensor(np.stack(np.where(adjacency_matrix == 1)), dtype=torch.long)
    x = torch.tensor(atom_features, dtype=torch.float)
    num_nodes = x.shape[0]  # Number of nodes in the current graph
    pos = torch.tensor(np.array(atom_positions), dtype=torch.float)
    y = torch.tensor(quadrupole, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    dataset.append(graph_data)

# Define the split sizes
total_samples = len(dataset)
train_size = 110000
val_size = 10000
test_size = total_samples - train_size - val_size

# Split the data using train_test_split with specific random seed
train_data, temp_data = train_test_split(dataset, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)

# Print the size of each split
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Test size:", len(test_data))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuadrupolePredictorEGNN(nn.Module):
    def __init__(self):
        super(QuadrupolePredictorEGNN, self).__init__()
        self.egnn = EGNN_Network(
            num_tokens=10,
            dim=128,
            depth=3
        )
        hidden_dim = 128
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 9)

    def forward(self, feats, coors, adj_mat):
        feats_out, _ = self.egnn(feats, coors, adj_mat=adj_mat)
        # Mean pooling across nodes, preserving batch and feature dims
        graph_embedding = feats_out.mean(dim=1)
        # Pass through fully connected layers
        graph_embedding = F.silu(self.fc1(graph_embedding))
        output = self.fc2(graph_embedding)
        # Reshape the output to (batch_size, 3, 3)
        output = output.view(-1, 3, 3)

        return output

net = QuadrupolePredictorEGNN()

net.to(device)
for name, value in vars(net).items():
    print(f'{name}: {value}')


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_function_l1 = nn.L1Loss()
best_val_loss = float('inf')
for epoch in range(1000):
    net.train()
    total_loss_l1 = 0
    total_loss_mse = 0
    total_r2 = 0
    for batch in train_loader:
        batch = batch.to(device)
        # Reshape the features and coordinates based on the batch vector
        feats, _ = to_dense_batch(batch.x, batch.batch) # Shape: (batch_size, num_nodes, num_features)
        feats = feats[:, :, 0:1]
        coors, _ = to_dense_batch(batch.pos, batch.batch) # Shape: (batch_size, num_nodes, 3)
        target = batch.y.view(-1, 3, 3)  # Shape: (batch_size, 3, 3) for quadrupole
        adj_mat = to_dense_adj(batch.edge_index, batch = batch.batch)
        feats_out = net(feats, coors, adj_mat=adj_mat)
        
        # Compute Losses
        loss_l1 = loss_function_l1(feats_out, target)
        loss_mse = mse_loss(feats_out, target)
        # Backward propagation
        loss_l1.backward()
        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()
        total_loss_l1 += loss_l1.item()
        total_loss_mse += loss_mse.item()
        r2 = r2_score(target.view(-1).cpu().numpy(), feats_out.view(-1).detach().cpu().numpy())
        total_r2 += r2

    avg_loss_l1 = total_loss_l1 / len(train_loader)
    avg_r2 = total_r2 / len(train_loader)
    avg_loss_mse = total_loss_mse / len(train_loader)
    print(f'Epoch {epoch}, L1 Loss: {avg_loss_l1}, MSE Loss: {avg_loss_mse}, R2 Score: {avg_r2}')
    
    # Validation
    net.eval()
    with torch.no_grad():
        val_loss_l1 = 0
        val_loss_mse = 0
        val_r2 = 0
        for batch in val_loader:
            batch = batch.to(device)
            feats, _ = to_dense_batch(batch.x, batch.batch)
            coors, _ = to_dense_batch(batch.pos, batch.batch)
            feats = feats[:, :, 0:1]
            target = batch.y.view(-1, 3, 3)  # Shape: (batch_size, 3, 3) for quadrupole
            adj_mat = to_dense_adj(batch.edge_index, batch = batch.batch)
            feats_out = net(feats, coors, adj_mat=adj_mat)

            # Compute Loss
            loss_l1 = loss_function_l1(feats_out, target)
            loss_mse = mse_loss(feats_out, target)
            val_loss_l1 += loss_l1.item()
            val_loss_mse += loss_mse.item()
            # Compute R2 score
            r2 = r2_score(target.view(-1).cpu().numpy(), feats_out.view(-1).detach().cpu().numpy())
            val_r2 += r2
        avg_val_loss_l1 = val_loss_l1 / len(val_loader)
        avg_val_loss_mse = val_loss_mse / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        print(f'Validation L1 Loss: {avg_val_loss_l1}, Validation MSE Loss: {avg_val_loss_mse}, Validation R2 Score: {avg_val_r2}')


        # Save the model if it has the best validation loss so far
        if avg_val_loss_l1 < best_val_loss:
            best_val_loss = avg_val_loss_l1
            torch.save(net.state_dict(), 'best_model_egnn_quadrupole_mae.pth')


# Testing
net.load_state_dict(torch.load('best_model_egnn_quadrupole_mae.pth'))
net.eval()
with torch.no_grad():
    test_loss_l1 = 0
    test_loss_mse = 0
    test_r2 = 0
    test_rmse = 0
    test_mape = 0
    test_evs = 0
    test_me = 0
    for batch in test_loader:
        batch = batch.to(device)
        feats, _ = to_dense_batch(batch.x, batch.batch)
        feats = feats[:, :, 0:1]
        coors, _ = to_dense_batch(batch.pos, batch.batch)
        target = batch.y.view(-1, 3, 3)  # Shape: (batch_size, 3, 3) for quadrupole
        adj_mat = to_dense_adj(batch.edge_index, batch = batch.batch)
        feats_out = net(feats, coors, adj_mat=adj_mat)

        # Compute Loss
        loss_l1 = loss_function_l1(feats_out, target)
        test_loss_l1 += loss_l1.item()
        loss_mse = mse_loss(feats_out, target)
        test_loss_mse += loss_mse.item()

        # Calculate additional metrics
        true_values = target.view(-1).cpu().numpy()
        pred_values = feats_out.view(-1).detach().cpu().numpy()
        r2 = r2_score(true_values, pred_values)
        test_r2 += r2
        test_rmse += np.sqrt(mean_squared_error(true_values, pred_values))
        test_mape += mean_absolute_percentage_error(true_values, pred_values)
        test_evs += explained_variance_score(true_values, pred_values)

    avg_test_loss_l1 = test_loss_l1 / len(test_loader)
    avg_test_r2 = test_r2 / len(test_loader)
    avg_test_loss_mse = test_loss_mse / len(test_loader)
    avg_test_rmse = test_rmse / len(test_loader)
    avg_test_mape = test_mape / len(test_loader)
    avg_test_evs = test_evs / len(test_loader)
    print(f'Test Loss L1: {avg_test_loss_l1}, MSE Loss: {avg_test_loss_mse}, Test R2 Score: {avg_test_r2}')
    print(f'Test RMSE: {avg_test_rmse}, MAPE: {avg_test_mape}, Explained Variance Score: {avg_test_evs}')
