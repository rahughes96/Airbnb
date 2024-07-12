import itertools
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import yaml
import json
import os
import datetime
import time

# Define numerical features and label
numerical_features = ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating',
                      'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count']
label = 'Price_Night'

class AirbnbNightlyPriceRegressionDataset(Dataset):

    def __init__(self, dataframe, scaler=None):
        """
        Initializes the AirbnbNightlyPriceRegressionDataset class by loading and normalizing features
        and labels from the provided dataframe.

        Parameters:
            dataframe (pd.DataFrame): DataFrame containing the dataset.
            scaler (StandardScaler, optional): StandardScaler object for feature normalization. If None, a new scaler is created.

        Returns:
            None
        """

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

        """
        Returns the length of the dataset.

        Parameters:
            None

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):

        """
        Retrieves a single sample from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features and label of the sample at the specified index.
        """

        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

class AirbnbPriceModel(nn.Module):
    def __init__(self, input_dim, config, dropout_prob):

        """
        Initializes the AirbnbPriceModel class by creating the neural network layers.

        Parameters:
            input_dim (int): The number of input features.
            config (dict): Configuration dictionary containing hidden_layer_width, depth, and dropout_prob.
            dropout_prob (float): Dropout probability.

        Returns:
            None
        """

        super(AirbnbPriceModel, self).__init__()
        hidden_layer_width = config['hidden_layer_width']
        depth = config['depth']
        dropout_prob = config.get('dropout_prob', 0.0)  # Default to 0.0 if not provided

        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))  # Add dropout after activation

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))  # Add dropout after activation

        layers.append(nn.Linear(hidden_layer_width, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):

        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor containing the features.

        Returns:
            torch.Tensor: Output tensor containing the predicted values.
        """

        return self.model(x)


def evaluate(model, dataloader, criterion):

    """
    Evaluates the model on the provided dataloader using the specified loss criterion.

    Parameters:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): DataLoader object providing the dataset.
        criterion (nn.Module): Loss function to use for evaluation.

    Returns:
        tuple: A tuple containing the average validation loss, root mean squared error, and R-squared score.
    """

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
    rmse = root_mean_squared_error(all_labels, all_outputs)
    r2 = r2_score(all_labels, all_outputs)
    return avg_val_loss, rmse, r2

def train(model, train_loader, val_loader, epochs, criterion, optimizer, writer, grad_clip=1.0, patience=5):

    """
    Trains the model for a specified number of epochs and logs the training and validation metrics.

    Parameters:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader object providing the training dataset.
        val_loader (DataLoader): DataLoader object providing the validation dataset.
        epochs (int): The number of epochs to train for.
        criterion (nn.Module): Loss function to use for training.
        optimizer (torch.optim.Optimizer): Optimizer to use for updating model parameters.
        writer (SummaryWriter): SummaryWriter object for logging metrics.
        grad_clip (float, optional): Gradient clipping value. Defaults to 1.0.

    Returns:
        float: The total duration of the training process.
    """

    start_time = time.time()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear gradients
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels.unsqueeze(1))  # Compute loss
            loss.backward()  # Backward pass
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()  # Update weights

            running_loss += loss.item()

            # Log weights and gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
                        writer.add_histogram(f'{name}.grad', param.grad.detach().cpu().numpy(), epoch)
                    else:
                        print(f"NaN or Inf detected in gradients of {name} at epoch {epoch}")
                if not torch.isnan(param).any() and not torch.isinf(param).any():
                    writer.add_histogram(f'{name}', param.detach().cpu().numpy(), epoch)
                else:
                    print(f"NaN or Inf detected in parameters of {name} at epoch {epoch}")

        avg_train_loss = running_loss / len(train_loader)
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()

        if epoch - best_epoch >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model_state)
    end_time = time.time()
    training_duration = end_time - start_time
    return training_duration


def save_model(model, config, metrics, model_path, best_model=False):

    """
    Saves the model's state dictionary, hyperparameters, and performance metrics to the specified directory.

    Parameters:
        model (nn.Module): The neural network model to save.
        config (dict): Configuration dictionary containing model hyperparameters.
        metrics (dict): Dictionary containing performance metrics.
        model_path (str): Directory path to save the model.
        best_model (bool, optional): Whether to save the model as the best model. Defaults to False.

    Returns:
        None
    """

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
    with open(os.path.join(model_path, 'hyperparameters.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Convert numpy float32 to standard float before saving to JSON
    def convert_metrics(metrics):
        if isinstance(metrics, dict):
            return {k: convert_metrics(v) for k, v in metrics.items()}
        elif isinstance(metrics, np.float32):
            return float(metrics)
        else:
            return metrics

    metrics = convert_metrics(metrics)

    with open(os.path.join(model_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    if best_model:
        best_model_path = os.path.join(model_path, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)

def calculate_inference_latency(model, dataloader, num_samples=100):

    """
    Calculates the average inference latency of the model on the provided dataloader.

    Parameters:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): DataLoader object providing the dataset.
        num_samples (int, optional): Number of samples to use for latency calculation. Defaults to 100.

    Returns:
        float: The average inference latency per sample.
    """

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

    """
    Generates a list of neural network configurations to try during model training.

    Parameters:
        None

    Returns:
        list: A list of dictionaries, each containing a unique set of hyperparameters for the model.
    """

    depths = [2, 3]  # Fixed at 2 layers for fine-tuning
    hidden_layer_widths = [256]  # Fixed at 128 units for fine-tuning
    learning_rates = [0.003, 0.005, 0.007]  # Different learning rates to try
    dropout_probs = [0.2]  # Different dropout probabilities to try
    weight_decays = [0.001]  # Different weight decays to try
    optimisers = ['Adam']  # Fixed optimiser for fine-tuning
    
    configs = []
    for depth, width, lr, dropout_prob, weight_decay, opt in itertools.product(depths, hidden_layer_widths, learning_rates, dropout_probs, weight_decays, optimisers):
        config = {
            'depth': depth,
            'hidden_layer_width': width,
            'optimiser': {
                'name': opt,
                'learning_rate': lr
            },
            'dropout_prob': dropout_prob,
            'weight_decay': weight_decay
        }
        configs.append(config)
        
    return configs[:16]  # Select the first 16 configurations


def find_best_nn(train_loader, val_loader, test_loader, epochs, writer):

    """
    Finds the best neural network configuration by training and evaluating models with different configurations.

    Parameters:
        train_loader (DataLoader): DataLoader object providing the training dataset.
        val_loader (DataLoader): DataLoader object providing the validation dataset.
        test_loader (DataLoader): DataLoader object providing the test dataset.
        epochs (int): The number of epochs to train each model.
        writer (SummaryWriter): SummaryWriter object for logging metrics.

    Returns:
        tuple: A tuple containing the best model, best metrics, and best configuration.
    """

    best_rmse = float('inf')
    best_model = None
    best_config = None
    best_metrics = None
    configs = generate_nn_configs()

    for config in configs:
        input_dim = len(numerical_features)
        model = AirbnbPriceModel(input_dim, config, dropout_prob=config['dropout_prob'])
        criterion = nn.MSELoss()
        optimizer = getattr(optim, config['optimiser']['name'])(model.parameters(), lr=config['optimiser']['learning_rate'], weight_decay=config['weight_decay'])

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

    save_model(best_model, best_config, best_metrics, 'models/neural_network/best_model', best_model=True)
    return best_model, best_metrics, best_config

if __name__ == "__main__":
    data = pd.read_csv('AirbnbData/Processed_Data/clean_tabular_data/clean_tabular_data.csv')
    dataset = AirbnbNightlyPriceRegressionDataset(data)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    writer = SummaryWriter()

    best_model, best_metrics, best_config = find_best_nn(train_loader, val_loader, test_loader, epochs=5, writer=writer)

    writer.close()

    print("Best Model Config:", best_config)
    print("Best Model Metrics:", best_metrics)