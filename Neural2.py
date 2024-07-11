import itertools
import torch
import yaml
import json
import os
import datetime
import time
import tabular_data
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score


# Define numerical features and label
numerical_features = ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating',
                      'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count']
label = 'Price_Night'

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, dataframe, scaler=None):
        self.dataframe = dataframe
        self.features = dataframe[numerical_features].astype(float).values
        self.labels = dataframe[label].values.astype(float)

        # Normalize the features
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

class AirbnbPriceModel(nn.Module):
    def __init__(self, input_dim, config):
        super(AirbnbPriceModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, config['hidden_layer_width']))
        layers.append(nn.ReLU())
        for _ in range(config['depth'] - 1):
            layers.append(nn.Linear(config['hidden_layer_width'], config['hidden_layer_width']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config['hidden_layer_width'], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def evaluate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
            all_labels.append(labels.numpy())
            all_outputs.append(outputs.numpy())
    avg_val_loss = val_loss / len(dataloader)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    # Changed mean_squared_error to root_mean_squared_error to avoid deprecation warning
    rmse = root_mean_squared_error(all_labels, all_outputs, squared=False)
    r2 = r2_score(all_labels, all_outputs)
    return avg_val_loss, rmse, r2

def train(model, train_loader, val_loader, epochs, criterion, optimizer, writer):
    start_time = time.time()
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear gradients
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels.unsqueeze(1))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

            # Log weights and gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad.detach().cpu().numpy(), epoch)
                writer.add_histogram(f'{name}', param.detach().cpu().numpy(), epoch)

        avg_train_loss = running_loss / len(train_loader)
        val_loss, _, _ = evaluate(model, val_loader, criterion)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    end_time = time.time()
    training_duration = end_time - start_time
    return training_duration

def convert_metrics(metrics):
    if isinstance(metrics, dict):
        return {k: convert_metrics(v) for k, v in metrics.items()}
    elif isinstance(metrics, np.float32):
        return float(metrics)
    else:
        return metrics

def save_model(model, config, metrics, model_path, best_model=False):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
    with open(os.path.join(model_path, 'hyperparameters.json'), 'w') as f:
        json.dump(config, f, indent=4)
    metrics = convert_metrics(metrics)
    with open(os.path.join(model_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    if best_model:
        best_model_path = os.path.join(model_path, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)

def calculate_inference_latency(model, dataloader, num_samples=100):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for i, (features, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            _ = model(features)
    end_time = time.time()
    avg_latency = (end_time - start_time) / num_samples
    return avg_latency

def generate_nn_configs():
    depths = [2, 3, 4, 5]
    hidden_layer_widths = [16, 32, 64, 128]
    learning_rates = [0.01, 0.001]
    optimisers = ['SGD', 'Adam']

    configs = []
    for depth, width, lr, opt in itertools.product(depths, hidden_layer_widths, learning_rates, optimisers):
        config = {
            'depth': depth,
            'hidden_layer_width': width,
            'optimiser': {
                'name': opt,
                'learning_rate': lr
            }
        }
        configs.append(config)
    return configs

def find_best_nn(train_loader, val_loader, test_loader, epochs, writer):
    best_rmse = float('inf')
    best_model = None
    best_config = None
    best_metrics = None
    configs = generate_nn_configs()

    for config in configs:
        input_dim = len(numerical_features)
        model = AirbnbPriceModel(input_dim, config)
        criterion = nn.MSELoss()
        optimizer = getattr(optim, config['optimiser']['name'])(model.parameters(), lr=config['optimiser']['learning_rate'])

        training_duration = train(model, train_loader, val_loader, epochs, criterion, optimizer, writer)

        _, train_rmse, train_r2 = evaluate(model, train_loader, criterion)
        _, val_rmse, val_r2 = evaluate(model, val_loader, criterion)
        _, test_rmse, test_r2 = evaluate(model, test_loader, criterion)

        inference_latency = calculate_inference_latency(model, test_loader)

        metrics = {
            'RMSE_loss': {
                'train': train_rmse,
                'validation': val_rmse,
                'test': test_rmse
            },
            'R_squared': {
                'train': train_r2,
                'validation': val_r2,
                'test': test_r2
            },
            'inference_latency': inference_latency,
            'training_duration': training_duration
        }

        save_model(model, config, metrics, f"models/neural_network/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model
            best_config = config
            best_metrics = metrics

    if best_model:
        save_model(best_model, best_config, best_metrics, "models/best_model", best_model=True)

    return best_model, best_metrics, best_config

if __name__ == "__main__":
    # Load your dataframe here
    df = pd.read_csv("AirbnbData/Processed_Data/clean_tabular_data/clean_tabular_data.csv")
    
    dataset = AirbnbNightlyPriceRegressionDataset(df)
    
    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    scaler = train_dataset.dataset.scaler
    test_dataset = AirbnbNightlyPriceRegressionDataset(test_dataset.dataset.dataframe.iloc[test_dataset.indices], scaler)

    # Further split the training dataset into training (80%) and validation (20%)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    writer = SummaryWriter()

    best_model, best_metrics, best_config = find_best_nn(train_loader, val_loader, test_loader, epochs=5, writer=writer)

    writer.close()

    print("Best Model Config:", best_config)
    print("Best Model Metrics:", best_metrics)




