import itertools
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tabular_data import map_Bedrooms
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, f1_score, recall_score, precision_score
import yaml
import json
import os
import datetime
import time

numerical_features = ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating',
                      'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating',
                      'amenities_count', 'Price_Night']  # Include Price_Night as a feature
label = 'bedrooms_mapped'  # New label is the number of bedrooms

class AirbnbNightlyPriceClassificationDataset(Dataset):
    """
    A dataset class to handle Airbnb data for classification tasks.
    
    Args:
        dataframe (pd.DataFrame): Input data containing the features and labels.
        scaler (StandardScaler, optional): A pre-fitted scaler to normalize features.
        
    Attributes:
        features (np.ndarray): The normalized feature set.
        labels (np.ndarray): The label set (number of bedrooms).
        scaler (StandardScaler): Scaler used to normalize the feature set.
    """
    def __init__(self, dataframe, scaler=None):
        self.dataframe = dataframe
        self.features = dataframe[numerical_features].astype(float).values
        self.labels = dataframe[label].values.astype(int)  # Ensure the label is an integer

        # Normalize the features
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            tuple: A tuple containing the features and the corresponding label.
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Classification label should be long/int
        return features, label

class AirbnbPriceClassificationModel(nn.Module):
    """
    A fully connected neural network model for classifying the number of bedrooms.
    
    Args:
        input_dim (int): Number of input features.
        config (dict): Configuration for the network, containing hidden layer width, depth, and dropout probability.
        dropout_prob (float): Probability of dropping out nodes during training.
        num_classes (int): Number of output classes (number of bedroom categories).
    
    Attributes:
        model (nn.Sequential): The neural network model.
    """
    def __init__(self, input_dim, config, dropout_prob, num_classes=5):
        super(AirbnbPriceClassificationModel, self).__init__()
        hidden_layer_width = config['hidden_layer_width']
        depth = config['depth']
        dropout_prob = config.get('dropout_prob', 0.0)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_layer_width, num_classes))  # Output for classification
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
    Evaluates the model on the given dataset.
    
    Args:
        model (nn.Module): The classification model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        criterion (Loss function): Loss function to use (CrossEntropyLoss).
    
    Returns:
        tuple: Average validation loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    accuracy_list = []
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Get predictions and compare with labels
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0) 

            alternate_accuracy = accuracy_score(labels, predicted)
            accuracy_list.append(alternate_accuracy)
            #from sklearn.metrics import accuracy_score

            #accuracy_score(labels,prediction)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    avg_val_loss = val_loss / len(dataloader)
    #accuracy = correct / total

    accuracy = sum(accuracy_list)/len(accuracy_list)

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)

    return avg_val_loss, accuracy, precision, recall, f1
"""
#print("previous_accuracy:", val_loss)
#print("alternate_accuracy:", alternate_accuracy)
return avg_val_loss, alternate_accuracy
"""

def train(model, train_loader, val_loader, epochs, criterion, optimizer, writer, grad_clip=1.0, patience=20):
    """
    Trains the classification model.
    
    Args:
        model (nn.Module): The classification model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        criterion (Loss function): Loss function (CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer to use during training.
        writer (SummaryWriter): TensorBoard SummaryWriter to log metrics.
        grad_clip (float): Value for gradient clipping.
        patience (int): Early stopping patience.
    
    Returns:
        float: Duration of training.
    """
    criterion = nn.CrossEntropyLoss()  # Classification loss function
    best_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0
    training_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss, _,_,_,_ = evaluate(model, val_loader, criterion)

        writer.add_scalars('Loss', {'train': avg_train_loss, 'validation': avg_val_loss}, epoch)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    training_duration = time.time() - training_start_time
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


def find_best_classification_nn(train_loader, val_loader, test_loader, epochs, writer):
    """
    Finds the best neural network model configuration by training on the dataset.
    
    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        epochs (int): Number of epochs to train.
        writer (SummaryWriter): TensorBoard SummaryWriter to log metrics.
    
    Returns:
        tuple: The best model, its metrics, and the configuration.
    """
    best_accuracy = 0.0
    best_model = None
    best_config = None
    best_metrics = None
    configs = generate_nn_configs()

    for config in configs:
        input_dim = len(numerical_features)
        model = AirbnbPriceClassificationModel(input_dim, config, dropout_prob=config['dropout_prob'], num_classes=9)
        criterion = nn.CrossEntropyLoss()
        optimizer = getattr(optim, config['optimiser']['name'])(model.parameters(), lr=config['optimiser']['learning_rate'], weight_decay=config['weight_decay'])

        training_duration = train(model, train_loader, val_loader, epochs, criterion, optimizer, writer)

        train_loss, train_acc, train_precision, train_recall,  train_f1 = evaluate(model, train_loader, criterion)
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion)
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion)

        metrics = {
            'Train': {
                'accuracy': train_acc,
                'precision': train_precision,
                'recall': train_recall,
                'f1': train_f1,
                'loss': train_loss
            },
            'Validation': {
                'accuracy': val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'loss': val_loss
            },
            'Test': {
                'accuracy': test_acc,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'loss': test_loss
            }
        }

        save_model(model, config, metrics, f"models/neural_network_classification/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

        #with open('classification/best_model/hyperparameters.json', 'r') as f:


        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_config = config
            best_metrics = metrics

    save_model(best_model, best_config, best_metrics, 'models/neural_network_classification/best_model', best_model=True)
    return best_model, best_metrics, best_config

if __name__ == "__main__":
    data = pd.read_csv('AirbnbData/Processed_Data/clean_tabular_data.csv')
    unique_bedrooms = sorted(data['bedrooms'].unique())
    bedroom_mapping = {bedroom: idx for idx, bedroom in enumerate(unique_bedrooms)}

    # Apply the mapping
    data['bedrooms_mapped'] = data['bedrooms'].map(bedroom_mapping)
    dataset = AirbnbNightlyPriceClassificationDataset(data)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    writer = SummaryWriter()

    best_model, best_metrics, best_config = find_best_classification_nn(train_loader, val_loader, test_loader, epochs=200, writer=writer)

    writer.close()

    print("Best Model Config:", best_config)
    print("Best Model Metrics:", best_metrics)