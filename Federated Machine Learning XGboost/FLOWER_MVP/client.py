# ---------------------------------------------
# Federated Learning Client with Flower (flwr)
# ---------------------------------------------
# This script builds a client for a federated learning setup.
# Each client trains a local model on its own subset of data
# and communicates with a central server to collaboratively train a model.

import flwr as fl  # Flower framework for federated learning
import torch  # PyTorch for building and training models
import sys  # Used to pass client ID from the command line
import torch.nn as nn  # Neural network building blocks
import numpy as np  # Used to split dataset by index
import torch.optim as optim  # Optimizer like SGD for updating weights
from torch.utils.data import DataLoader, Dataset  # For custom dataset loading and batching
import pandas as pd  # To load and manipulate the CSV data
from sklearn.model_selection import train_test_split  # For train/test split
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding labels and scaling features
from sklearn.metrics import precision_score, recall_score, f1_score  # Metrics for model evaluation

# ---------------------------------------------
# Custom Dataset class to work with PyTorch
# ---------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, df, label_col):
        self.y = torch.tensor(df[label_col].values, dtype=torch.float32).view(-1, 1)  # Convert label column to a tensor
        self.X = torch.tensor(df.drop(columns=[label_col]).values, dtype=torch.float32)  # Convert features to tensor

    def __len__(self):
        return len(self.X)  # Total number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Retrieve one sample (features and label)

# ---------------------------------------------
# A simple Logistic Regression model for binary classification
# ---------------------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer maps input_dim -> 1 output

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Output a probability between 0 and 1

# ---------------------------------------------
# Function to load and preprocess the dataset
# ---------------------------------------------
def load_data(csv_path, label_col="label"):
    df = pd.read_csv(csv_path)  # Load dataset from CSV file
    df = df.drop(columns=["type"])  # Drop column that is not needed

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])  # Convert categorical values to numeric

    scaler = StandardScaler()  # Create a scaler to normalize feature values
    df[df.columns.difference([label_col])] = scaler.fit_transform(df[df.columns.difference([label_col])])

    train_df, _ = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)  # Optional: retain only training set
    return CustomDataset(train_df, label_col)  # Return a PyTorch dataset

# ---------------------------------------------
# Split the full dataset into equal-sized partitions for each client
# ---------------------------------------------
def split_dataset(df, label_col, num_clients):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataset
    indices = np.array_split(df.index, num_clients)  # Create equal splits by index

    partitions = []
    for idx_chunk in indices:
        part_df = df.loc[idx_chunk].reset_index(drop=True)  # Extract partitioned data
        partitions.append(CustomDataset(part_df, label_col))  # Convert each chunk to PyTorch dataset

    return partitions  # Return list of datasets (1 per client)

# ---------------------------------------------
# Flower Client definition
# ---------------------------------------------
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, dataset):
        self.model = model  # Local model instance
        self.dataset = dataset  # Local data
        self.loader = DataLoader(dataset, batch_size=16, shuffle=True)  # DataLoader to iterate over local data
        self.criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification

    def get_parameters(self, config):
        return [val.detach().cpu().numpy().copy() for val in self.model.state_dict().values()]  # Send model parameters to server

    def set_parameters(self, parameters):
        # Receive parameters from server and load them into local model
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("Client: Starting fit()")
        try:
            self.set_parameters(parameters)  # Set global parameters into local model
            optimizer = optim.SGD(self.model.parameters(), lr=0.1)  # Initialize optimizer
            self.model.train()

            for x, y in self.loader:
                optimizer.zero_grad()  # Clear gradients
                pred = self.model(x)  # Forward pass
                loss = self.criterion(pred, y)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

            updated_params = self.get_parameters(config={})  # Get updated weights
            num_examples = int(len(self.dataset))  # Number of training samples
            metrics = {"train_loss": float(loss.item())}  # Report last batch's loss

            print("Client: Finished fit() with:")
            print(f" - parameters: {len(updated_params)} tensors")
            print(f" - examples: {num_examples}")
            print(f" - metrics: {metrics}")

            return updated_params, num_examples, metrics  # Return to server

        except Exception as e:
            print("Client: Exception during fit():", e)
            raise e

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)  # Set received weights
        self.model.eval()  # Set model to evaluation mode

        loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # No gradient tracking
            for x, y in self.loader:
                pred = self.model(x)  # Get predictions
                loss += self.criterion(pred, y).item()  # Accumulate loss
                predicted = (pred > 0.5).float()  # Convert probabilities to 0 or 1
                all_preds.extend(predicted.view(-1).cpu().numpy())
                all_labels.extend(y.view(-1).cpu().numpy())
                correct += (predicted == y).sum().item()
                total += y.size(0)

        # Compute evaluation metrics
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"Eval Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        return float(loss) / len(self.loader), int(len(self.dataset)), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

# ---------------------------------------------
# Main script execution (entry point)
# ---------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("Train_Test_IoT_Fridge.csv")  # Load data from CSV
    df = df.drop(columns=["type"])  # Drop irrelevant column

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])  # Encode categorical features

    scaler = StandardScaler()
    df[df.columns.difference(["label"])] = scaler.fit_transform(df[df.columns.difference(["label"])])

    num_clients = 2  # Total number of clients
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Read client ID from command line

    clients_data = split_dataset(df, label_col="label", num_clients=num_clients)  # Split dataset among clients
    dataset = clients_data[client_id]  # This client's dataset

    input_dim = dataset[0][0].shape[0]  # Number of input features
    model = LogisticRegressionModel(input_dim)  # Initialize model

    fl.client.start_client(
        server_address="localhost:8080",  # Address of the Flower server
        client=FLClient(model, dataset).to_client()  # Create and start the client
    )