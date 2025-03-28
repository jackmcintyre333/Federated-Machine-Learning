# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # For model evaluation
import xgboost as xgb  # For gradient boosting implementation
from typing import List, Tuple  # For type hinting
import logging  # For logging system events and debugging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate log filename with timestamp
log_filename = f"logs/federated_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging settings for both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # File handler
        logging.StreamHandler()             # Console handler
    ]
)
logger = logging.getLogger(__name__)

class FederatedXGBoost:
    """
    Implementation of Federated Learning using XGBoost for IoT intrusion detection.
    This class handles the distribution of data among clients, local model training,
    and aggregation of predictions from multiple models.
    """
    
    def __init__(self, n_clients: int = 3):
        """
        Initialize the federated learning system.
        
        Args:
            n_clients (int): Number of clients participating in federated learning
        """
        logger.info("Initializing FederatedXGBoost")
        logger.info(f"Number of clients: {n_clients}")
        self.n_clients = n_clients  # Number of participating clients
        self.models = []  # List to store individual client models
        self.label_encoder = LabelEncoder()  # For encoding categorical variables
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data by encoding categorical variables and splitting features and target.
        
        Args:
            data (pd.DataFrame): Raw input data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed features (X) and target (y)
        """
        logger.info("Starting data preprocessing")
        logger.info(f"Initial data shape: {data.shape}")
        
        # Extract features and target, assuming last two columns are special
        X = data.iloc[:, :-2]  # All columns except last two
        y = data.iloc[:, -2]   # Second-to-last column as target
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        logger.info(f"Found {len(categorical_columns)} categorical columns")
        
        # Encode all categorical variables using LabelEncoder
        for column in categorical_columns:
            logger.debug(f"Encoding column: {column}")
            X[column] = self.label_encoder.fit_transform(X[column])
        
        # Encode target variable
        y = self.label_encoder.fit_transform(y)
        
        # Convert pandas objects to numpy arrays for better performance
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        
        logger.info(f"Number of unique classes: {len(np.unique(y))}")
        logger.info(f"Preprocessed X shape: {X.shape}")
        logger.info(f"Preprocessed y shape: {y.shape}")
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets while maintaining class distribution.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            test_size (float): Proportion of dataset to include in test split
            
        Returns:
            Tuple containing train-test split of inputs and targets
        """
        logger.info(f"Splitting data with test_size={test_size}")
        result = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        logger.info(f"Training set size: {result[0].shape[0]}")
        logger.info(f"Test set size: {result[1].shape[0]}")
        return result
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Distribute data among clients in a federated setting.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            List[Tuple]: List of (X, y) tuples for each client
        """
        logger.info("Distributing data among clients")
        client_data = []
        data_size = len(X)
        chunk_size = data_size // self.n_clients  # Calculate size for each client
        
        # Distribute data among clients
        for i in range(self.n_clients):
            start_idx = i * chunk_size
            # Handle last chunk to include remaining data
            end_idx = start_idx + chunk_size if i < self.n_clients - 1 else data_size
            client_data.append((X[start_idx:end_idx], y[start_idx:end_idx]))
            logger.info(f"Client {i+1} data size: {end_idx - start_idx}")
        
        return client_data
    
    def train_local_models(self, client_data: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Train individual XGBoost models on each client's local data.
        
        Args:
            client_data (List[Tuple]): List of (X, y) data tuples for each client
        """
        logger.info("Starting local model training")
        for i, (X_client, y_client) in enumerate(client_data):
            logger.info(f"Training model for client {i+1}")
            logger.info(f"Client {i+1} training data shape: {X_client.shape}")
            
            start_time = datetime.now()
            # Initialize XGBoost classifier for multi-class classification
            model = xgb.XGBClassifier(
                objective='multi:softmax',  # Multi-class classification
                num_class=len(np.unique(y_client)),  # Number of unique classes
                random_state=42  # For reproducibility
            )
            # Train model on client's local data
            model.fit(X_client, y_client)
            training_time = datetime.now() - start_time
            
            self.models.append(model)
            logger.info(f"Client {i+1} model trained successfully in {training_time.total_seconds():.2f} seconds")
    
    def aggregate_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions from all client models using majority voting.
        
        Args:
            X (np.ndarray): Features to predict
            
        Returns:
            np.ndarray: Final aggregated predictions
        """
        logger.info("Aggregating predictions from all models")
        # Initialize array to store predictions from each model
        predictions = np.zeros((len(X), len(self.models)))
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            logger.debug(f"Getting predictions from model {i+1}")
            predictions[:, i] = model.predict(X)
        
        # Perform majority voting across all model predictions
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 1, predictions
        )
        logger.info("Prediction aggregation completed")
        return final_predictions
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            Tuple: (accuracy, precision, recall, f1) scores
        """
        logger.info("Evaluating model performance")
        
        # Calculate various performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Log the results
        logger.info("=== Model Performance Metrics ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("==============================")
        
        return accuracy, precision, recall, f1

def main():
    """
    Main execution function that orchestrates the federated learning process.
    """
    logger.info("=== Starting Federated Learning Process ===")
    start_time = datetime.now()
    
    try:
        logger.info("Loading dataset")
        # Load the IoT Fridge dataset
        data_path = "Train_Test_IoT_Fridge.csv"
        data = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully: {data.shape}")
        
        # Initialize federated learning system with 3 clients
        federated_model = FederatedXGBoost(n_clients=3)
        
        # Preprocess the raw data
        X, y = federated_model.preprocess_data(data)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = federated_model.split_data(X, y)
        
        # Distribute training data among clients
        client_data = federated_model.distribute_data(X_train, y_train)
        
        # Train local models on each client
        federated_model.train_local_models(client_data)
        
        # Generate predictions using aggregated model
        y_pred = federated_model.aggregate_predictions(X_test)
        
        # Evaluate the federated model
        federated_model.evaluate(y_test, y_pred)
        
        total_time = datetime.now() - start_time
        logger.info(f"=== Federated Learning Process Completed ===")
        logger.info(f"Total execution time: {total_time.total_seconds():.2f} seconds")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

# Entry point of the script
if __name__ == "__main__":
    main() 