import flwr as fl
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from typing import Tuple, Dict, List
import logging
import os
from datetime import datetime
import time
import json
import pickle
import tempfile
import atexit

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging to both file and console
log_filename = f"logs/client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XGBoostClient(fl.client.NumPyClient):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        logger.info("Initializing XGBoostClient")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.label_encoder = LabelEncoder()
        self.round_start_time = None
        self.temp_files = []  # Keep track of temporary files
        atexit.register(self._cleanup_temp_files)  # Register cleanup function
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Number of unique classes: {len(np.unique(y_train))}")
    
    def _cleanup_temp_files(self):
        """Clean up any remaining temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
        
    def _create_model(self):
        """Create and initialize a new XGBoost model."""
        logger.info("Creating new XGBoost model")
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(self.y_train)),
            random_state=42
        )
        # Initialize the model with a dummy fit
        self.model.fit(self.X_train, self.y_train, verbose=False)
        return self.model
        
    def get_parameters(self, config):
        """Get model parameters."""
        logger.debug("Getting model parameters")
        if self.model is None:
            self._create_model()
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        temp_file.close()
        self.temp_files.append(temp_file.name)
        
        try:
            # Save model to the temporary file
            self.model.save_model(temp_file.name)
            # Read the saved model as bytes
            with open(temp_file.name, 'rb') as f:
                model_bytes = f.read()
        finally:
            # Clean up the temporary file
            try:
                os.remove(temp_file.name)
                self.temp_files.remove(temp_file.name)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file.name}: {e}")
        
        # Convert to numpy array
        return [np.frombuffer(model_bytes, dtype=np.uint8)]
    
    def fit(self, parameters, config):
        """Train the model on local data."""
        if self.round_start_time is None:
            self.round_start_time = time.time()
            logger.info(f"Starting local training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("Starting local training")
        
        # Create new model if needed
        if self.model is None:
            self._create_model()
        
        # Update model parameters if provided
        if parameters:
            logger.debug("Updating model parameters")
            try:
                # Convert numpy array back to bytes
                model_bytes = parameters[0].tobytes()
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
                temp_file.close()
                self.temp_files.append(temp_file.name)
                
                try:
                    # Write model bytes to temporary file
                    with open(temp_file.name, 'wb') as f:
                        f.write(model_bytes)
                    # Load the model
                    self.model.load_model(temp_file.name)
                finally:
                    # Clean up the temporary file
                    try:
                        os.remove(temp_file.name)
                        self.temp_files.remove(temp_file.name)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file {temp_file.name}: {e}")
            except Exception as e:
                logger.error(f"Error loading model parameters: {e}")
                # If parameter loading fails, continue with current model
        
        # Train the model
        logger.info("Training model on local data")
        self.model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        training_duration = time.time() - self.round_start_time
        logger.info(f"Local training completed after {training_duration:.2f} seconds")
        self.round_start_time = None
        
        # Return parameters in the format expected by Flower
        return self.get_parameters(config), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local data."""
        if self.round_start_time is None:
            self.round_start_time = time.time()
            logger.info(f"Starting local evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("Starting local evaluation")
        
        # Create new model if needed
        if self.model is None:
            self._create_model()
        
        # Update model parameters if provided
        if parameters:
            logger.debug("Updating model parameters")
            try:
                # Convert numpy array back to bytes
                model_bytes = parameters[0].tobytes()
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
                temp_file.close()
                self.temp_files.append(temp_file.name)
                
                try:
                    # Write model bytes to temporary file
                    with open(temp_file.name, 'wb') as f:
                        f.write(model_bytes)
                    # Load the model
                    self.model.load_model(temp_file.name)
                finally:
                    # Clean up the temporary file
                    try:
                        os.remove(temp_file.name)
                        self.temp_files.remove(temp_file.name)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary file {temp_file.name}: {e}")
            except Exception as e:
                logger.error(f"Error loading model parameters: {e}")
                # If parameter loading fails, continue with current model
        
        # Make predictions
        logger.debug("Making predictions on test set")
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics")
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        evaluation_duration = time.time() - self.round_start_time
        logger.info(f"Local evaluation completed after {evaluation_duration:.2f} seconds")
        logger.info(f"Local evaluation results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        self.round_start_time = None
        return float(1 - accuracy), len(self.X_test), metrics

def load_data(client_id: int = 0):
    """Load and preprocess the data with different splits for each client."""
    logger.info(f"Loading and preprocessing data for client {client_id}")
    start_time = time.time()
    
    # Load data
    data = pd.read_csv("Train_Test_IoT_Fridge.csv")
    logger.info(f"Loaded data shape: {data.shape}")
    
    # Separate features and target
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = X.select_dtypes(include=['object']).columns
    logger.info(f"Found {len(categorical_columns)} categorical columns")
    
    for column in categorical_columns:
        logger.debug(f"Encoding column: {column}")
        X[column] = label_encoder.fit_transform(X[column])
    
    # Encode target variable
    logger.debug("Encoding target variable")
    y = label_encoder.fit_transform(y)
    
    # Create different splits for each client
    # Use client_id as part of the random_state to ensure different splits
    random_state = 42 + client_id
    
    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    
    # Second split: 80% of temp for test, 20% for validation
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.8, stratify=y_temp, random_state=random_state
    )
    
    # Log class distribution
    logger.info(f"Client {client_id} class distribution:")
    logger.info(f"Train set: {np.bincount(y_train)}")
    logger.info(f"Test set: {np.bincount(y_test)}")
    logger.info(f"Validation set: {np.bincount(y_val)}")
    
    data_loading_duration = time.time() - start_time
    logger.info(f"Data loading and preprocessing completed after {data_loading_duration:.2f} seconds")
    
    return X_train, X_test, y_train, y_test

def main():
    logger.info("Starting client")
    start_time = time.time()
    
    # Get client ID from environment variable or default to 0
    client_id = int(os.getenv('CLIENT_ID', '0'))
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data(client_id)
    
    # Create client
    client = XGBoostClient(X_train, y_train, X_test, y_test)
    
    # Start Flower client using the new method
    logger.info("Connecting to server at 127.0.0.1:8080")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )
    
    total_duration = time.time() - start_time
    logger.info(f"Client finished after {total_duration:.2f} seconds")

if __name__ == "__main__":
    main() 