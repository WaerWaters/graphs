import argparse
import contextlib
import datetime
import json
import math
import multiprocessing
import os
import time
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, GCNConv, GraphConv, MessagePassing
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import contextmanager


# Initialize CUDA
torch.cuda.init()
N_GPUS = torch.cuda.device_count()
BATCH_SIZE = 2024*32

# Check available GPUs
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

data_folder_path = ""
nodes = pd.read_csv(os.path.join(data_folder_path, "nodes.csv"))
edges = pd.read_csv(os.path.join(data_folder_path, "edges.csv"))
edge_attributes = pd.read_csv(os.path.join(data_folder_path, "edges_attributes.csv"))
node_split = pd.read_csv(os.path.join(data_folder_path, "splits.csv"))
pos = pd.read_csv(os.path.join(data_folder_path, "pos.csv"))

# edges features were not used.
# pos includes the x and y coordinates of the nodes
edge_attributes["lanes"] = edge_attributes["lanes"].astype(float)
edge_attributes["oneway"] = edge_attributes["oneway"].astype(float)
# fill maxspeed nan values with 0
edge_attributes["maxspeed"] = edge_attributes["maxspeed"].fillna(0).astype(float)

# Min-Max normalize maxspeed
edge_attributes["maxspeed"] = (edge_attributes["maxspeed"] - edge_attributes["maxspeed"].min()) / (edge_attributes["maxspeed"].max() - edge_attributes["maxspeed"].min())
# Min-Max normalize the attributes length and lanes
edge_attributes["length"] = (edge_attributes["length"] - edge_attributes["length"].min()) / (edge_attributes["length"].max() - edge_attributes["length"].min())
edge_attributes["lanes"] = (edge_attributes["lanes"] - edge_attributes["lanes"].min()) / (edge_attributes["lanes"].max() - edge_attributes["lanes"].min())


attributes = nodes.drop(columns=["accident_score"])
target = nodes["accident_score"]

x = torch.tensor(attributes.values, dtype=torch.float)
y = torch.tensor(target.values, dtype=torch.float)
edges = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attributes.values, dtype=torch.float)

pos = torch.tensor(pos.values, dtype=torch.float)  # assuming pos["x"] and pos["y"] are the columns in pos DataFrame

dataset = Data(
    x=x, y=y, edge_index=edges, edge_attr=edge_attr, pos=pos,
    test_mask=torch.tensor(node_split["test"].values, dtype=torch.bool),
    val_mask=torch.tensor(node_split["validation"].values, dtype=torch.bool),
    train_mask=torch.tensor(node_split["train"].values, dtype=torch.bool)
)

def append_to_json_file(filename, metrics): # Append the metrics to the log file
    # Check if the file exists
    if os.path.exists(filename):
        # If the file exists, open it and load the data
        with open(filename, 'r') as f:
            try:
                existing_data = json.load(f)
            except ValueError:
                existing_data = []
    else:
        # If the file doesn't exist, create it and initialize the data as an empty list
        existing_data = []

    # Append the new data
    existing_data.append(metrics)

    # Write everything back to the file
    with open(filename, 'w') as f:
        json.dump(existing_data, f)






def define_model(trial, model_type, layers_for_model):

    in_features = dataset.num_features

    if model_type == 'gcn':
        class GCN(nn.Module):
            def __init__(self, in_features, layers_for_model):
                super(GCN, self).__init__()
                self.layers = nn.ModuleList()
                for i in range(layers_for_model):
                    out_features = trial.suggest_int(f"n_of_units_layer_{i}", 4, 128)
                    dropout = trial.suggest_float(f"dropout_layer_{i}", 0.2, 0.5)
                    self.layers += [GCNConv(in_features, out_features), nn.ReLU(), nn.Dropout(dropout)]
                    in_features = out_features
                self.lin = nn.Linear(in_features, 1)
    
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                for layer in self.layers:
                    if isinstance(layer, GCNConv):
                        x = layer(x, edge_index)
                    else:
                        x = layer(x)
                x = self.lin(x)
                return x.squeeze(1)
        return GCN(in_features, layers_for_model)

    elif model_type == 'gnn':
        class GNNModel(nn.Module):
            def __init__(self, in_features, layers_for_model):
                super(GNNModel, self).__init__()
                self.layers = nn.ModuleList()
                for i in range(layers_for_model):
                    out_features = trial.suggest_int(f"n_of_units_layer_{i}", 4, 128)
                    dropout = trial.suggest_float(f"dropout_layer_{i}", 0.2, 0.5)
                    self.layers += [GraphConv(in_features, out_features), nn.ReLU(), nn.Dropout(dropout)]
                    in_features = out_features
                self.layers.append(GraphConv(in_features, 1))

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                for layer in self.layers:
                    if isinstance(layer, GraphConv):  # Only this layer needs edge_index
                        x = layer(x, edge_index)
                    else:
                        x = layer(x)
                return x.squeeze()
        return GNNModel(in_features, layers_for_model)

    elif model_type == 'gat':
        class GAT(nn.Module):
            def __init__(self, in_features, layers_for_model):
                super(GAT, self).__init__()
                self.layers = nn.ModuleList()
                for i in range(layers_for_model):
                    out_features = trial.suggest_int(f"n_of_units_layer_{i}", 4, 128)
                    dropout = trial.suggest_float(f"dropout_layer_{i}", 0.2, 0.5)
                    heads = trial.suggest_int(f"heads_layer_{i}", 1, 8)
                    self.layers.append(GATConv(in_features, out_features, heads=heads, dropout=dropout))
                    self.layers.append(nn.ReLU())
                    self.layers.append(nn.Dropout(dropout))
                    in_features = out_features * heads
                self.lin = nn.Linear(in_features, 1)
                self.leaky_relu = nn.LeakyReLU(0.1)

            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                for layer in self.layers:
                    if isinstance(layer, GATConv):
                        x = layer(x, edge_index)
                    else:
                        x = layer(x)
                x = self.lin(x)
                x = self.leaky_relu(x)
                return x.squeeze()
        return GAT(in_features, layers_for_model)
    
        


def train_and_evaluate(model, optimizer, train_loader, val_loader, device, params, epochs):
    train_losses = []
    metrics = {
        "device": str(device),
        "params": params,
        "epoch_losses_train": [],
        "train_losses": [],
        "val_losses": [],
        "total_train_loss": 0,
        "total_val_loss": 0
    }

    # Move the model to the specified device
    model.to(device)
    model.train()
    total_loss = 0
    for j in range(epochs):
        print("Device", device, "Epoch: ", j + 1, "/", epochs, "Progress: ", round(j / epochs, 2))
        train_losses = []
        for i, data in enumerate(train_loader, 1):
            
            data = data.to(device) # Move data to the specified device
            optimizer.zero_grad() # Zero the gradients before running the backward pass to avoid accumulating them on every iteration
            output = model(data)
            loss = F.mse_loss(output.squeeze(), data.y)
            total_loss += loss.item()
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if math.isnan(float(loss.item())): # Check if the loss is NaN and break the loop if it is
                break
        metrics['epoch_losses_train'].append(train_losses)

    average_loss = total_loss / len(train_loader)
    metrics["total_train_loss"] = total_loss
    metrics["average_train_loss"] = average_loss
    print("AVG Loss Train: ", average_loss)

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            # Move data to the specified device
            data = data.to(device)
            output = model(data)
            loss = F.mse_loss(output.squeeze(), data.y)
            total_loss += loss.item()
            metrics["val_losses"].append(loss.item())
            if math.isnan(float(loss.item())): # Check if the loss is NaN and break the loop if it is
                break

    average_loss_val = total_loss / len(val_loader)
    metrics["total_val_loss"] = total_loss
    metrics["average_val_loss"] = average_loss_val
    print("AVG Loss Val: ", average_loss_val)
    
    # Append the new data to the log JSON file
    append_to_json_file(filename, metrics)

    return average_loss_val


class GpuQueue: # thanks to Marc Schwering(https://vordeck.de/kn/optuna-gpu-queue) for inspiration on this class.
    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)

class Objective:
    def __init__(self, gpu_queue: GpuQueue, model_type: str):
        self.gpu_queue = gpu_queue
        self.model_type = model_type

    def __call__(self, trial):
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            print("Trial: ", trial.number, "GPU: ", gpu_i)
            print(trial.params)
            best_val_loss = objective(trial, gpu_i, self.model_type)
            return best_val_loss

def objective(trial, gpu_i, model_type):
    # Set device to GPU
    device = torch.device(f"cuda:{gpu_i}")

    # Define the number of layers for the model
    layers_for_model = trial.suggest_int("n_layers", 1, 20)

    # Define the model
    model = define_model(trial, model_type, layers_for_model)
    
    # Define the optimizer using lr and weight_decay
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "AdamW"])
    lr = trial.suggest_float("lr", 0.00001, 0.05)
    weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.05)
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define the number of epochs
    epochs = trial.suggest_int("epochs", 1, 100)
    
    # Define the train loader
    train_loader = NeighborLoader(
        dataset,
        num_neighbors=[10] * layers_for_model,
        input_nodes=dataset.train_mask,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Define the validation loader
    val_loader = NeighborLoader(
        dataset,
        num_neighbors=[10] * layers_for_model,
        input_nodes=dataset.val_mask,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Train and evaluate the model, and get the average loss
    validation_loss = train_and_evaluate(model, optimizer, train_loader, val_loader, device, trial.params, epochs)

    # Return the average loss
    return validation_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify the model type for training.")
    parser.add_argument('--model_type', type=str, choices=['gcn', 'gat', 'gnn'], required=True, help="Model type: 'gcn', 'gat', or 'gnn'")
    model_type = parser.parse_args()
    time_name = int(time.time())
    filename = f'metrics_w_edges_{time_name}_{model_type}.json' # Log json file in case of crash   
    study_name = f'metrics_{model_type}_{time_name}'  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name)
    study.optimize(Objective(GpuQueue(), model_type=model_type), n_trials=100, n_jobs=N_GPUS)
    print("Best trial:", study.best_trial.params)
    df = study.trials_dataframe()
    # Save the results to a csv file, 
    df.to_csv(f'optuna_results_w_edge_{model_type}_{datetime.datetime.now().strftime("%Y-%m-%d")}_{time_name}.csv')
    print("Best trial:", study.best_trial.params)