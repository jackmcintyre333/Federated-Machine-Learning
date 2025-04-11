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
import gc
import psutil
import sys
import traceback

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging to both file and console
log_filename = f"logs/client_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def log_exception(e: Exception):
    """Log exception with traceback."""
    logger.error(f"Exception occurred: {str(e)}")
    logger.error("Traceback:")
    logger.error(traceback.format_exc())

class XGBoostClient(fl.client.NumPyClient):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        logger.info("Initializing XGBoostClient")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # Initialize with smaller model
        self.model = self._create_model()
        self.round_start_time = None
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        log_memory_usage()

    def _create_model(self):
        """Create a new XGBoost model with minimal fixed structure."""
        logger.info("Creating new XGBoost model with minimal structure")
        
        try:
            # Use a very simple model configuration with fixed parameters
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=1,            # Minimal max depth
                learning_rate=0.1,
                n_estimators=3,         # Very few trees
                tree_method='hist',     # Use histogram-based algorithm
                random_state=42,
                max_bin=8,             # Very few bins
                subsample=0.5,         # Use only half the data
                colsample_bytree=0.5,  # Use only half the features
                min_child_weight=1,
                gamma=0,
                reg_alpha=0,
                reg_lambda=1,
                # Force CPU usage
                gpu_id=-1,             # Disable GPU
                predictor='cpu_predictor',  # Force CPU predictor
                n_jobs=1               # Use single thread
            )
            
            # Create a fixed-size initial model
            X_init = np.zeros((5, self.X_train.shape[1]))
            y_init = np.zeros(5)
            
            # Train on this fixed dataset to create a consistent model structure
            logger.debug("Training initial model")
            model.fit(X_init, y_init, verbose=False)
            logger.info("Initial model created successfully")
            log_memory_usage()
            return model
            
        except Exception as e:
            log_exception(e)
            logger.warning("Creating fallback minimal model")
            # Create a dummy model if fitting fails
            model = xgb.XGBClassifier(
                objective='binary:logistic', 
                random_state=42,
                gpu_id=-1,
                predictor='cpu_predictor',
                n_jobs=1
            )
            model.fit(np.array([[0]]), np.array([0]), verbose=False)
            return model

    def get_parameters(self, config):
        """Get model parameters."""
        logger.debug("Getting model parameters")
        log_memory_usage()
        
        if self.model is None:
            self.model = self._create_model()
        
        try:
            # Get model bytes directly from booster
            model_bytes = self.model.get_booster().save_raw()
            logger.info(f"Model parameter size: {len(model_bytes)} bytes")
            return [np.frombuffer(model_bytes, dtype=np.uint8)]
        except Exception as e:
            log_exception(e)
            logger.error("Returning empty parameters")
            return [np.array([], dtype=np.uint8)]

    def fit(self, parameters, config):
        """Train the model on local data."""
        logger.debug("Starting fit operation")
        log_memory_usage()
        
        if self.round_start_time is None:
            self.round_start_time = time.time()
            logger.info(f"Starting local training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update model parameters if provided
        if parameters:
            try:
                # Create a new model with the same structure
                self.model = self._create_model()
                
                # Load the parameters
                logger.debug("Loading provided model parameters")
                booster = xgb.Booster()
                booster.load_model(bytearray(parameters[0].tobytes()))
                self.model._Booster = booster
                logger.info("Loaded model parameters successfully")
                log_memory_usage()
            except Exception as e:
                log_exception(e)
                logger.warning("Using new model instead")
                self.model = self._create_model()
        
        # Train for one round with reduced data
        try:
            # Use a subset of data for training
            sample_size = min(100, len(self.X_train))  # Reduced from 500 to 100
            logger.debug(f"Using sample size of {sample_size} for training")
            X_sample = self.X_train[:sample_size]
            y_sample = self.y_train[:sample_size]
            
            # Convert to DMatrix for memory efficiency
            logger.debug("Converting to DMatrix")
            dtrain = xgb.DMatrix(X_sample, label=y_sample)
            
            # Update for just one round
            logger.debug("Updating model for one round")
            self.model._Booster.update(dtrain, self.model._Booster.num_boosted_rounds())
            
            # Log training metrics
            logger.debug("Computing training metrics")
            y_pred = self.model.predict(X_sample)
            train_acc = accuracy_score(y_sample, (y_pred > 0.5))
            logger.info(f"Training accuracy: {train_acc:.4f}")
            
            # Clean up
            del dtrain, X_sample, y_sample, y_pred
            gc.collect()
            log_memory_usage()
            
        except Exception as e:
            log_exception(e)
            logger.warning("Training failed, creating new model")
            self.model = self._create_model()
        
        training_duration = time.time() - self.round_start_time
        logger.info(f"Local training completed after {training_duration:.2f} seconds")
        self.round_start_time = None
        
        # Get and log model size
        params = self.get_parameters(config)
        logger.info(f"Model parameter size after training: {len(params[0])} bytes")
        log_memory_usage()
        
        return params, len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on local data."""
        logger.debug("Starting evaluation")
        log_memory_usage()
        
        if parameters:
            try:
                # Create a new model with the same structure
                self.model = self._create_model()
                
                # Load the parameters
                logger.debug("Loading model parameters for evaluation")
                booster = xgb.Booster()
                booster.load_model(bytearray(parameters[0].tobytes()))
                self.model._Booster = booster
                logger.info("Loaded model parameters for evaluation successfully")
                log_memory_usage()
            except Exception as e:
                log_exception(e)
                logger.warning("Using current model for evaluation")
        
        try:
            # Use a subset of test data for evaluation
            sample_size = min(100, len(self.X_test))  # Reduced from 500 to 100
            logger.debug(f"Using sample size of {sample_size} for evaluation")
            X_test_sample = self.X_test[:sample_size]
            y_test_sample = self.y_test[:sample_size]
            
            # Make predictions
            logger.debug("Making predictions")
            y_pred = self.model.predict(X_test_sample)
            y_pred_binary = (y_pred > 0.5)
            
            # Calculate metrics
            logger.debug("Computing evaluation metrics")
            metrics = {
                "accuracy": float(accuracy_score(y_test_sample, y_pred_binary)),
                "precision": float(precision_score(y_test_sample, y_pred_binary, average='binary')),
                "recall": float(recall_score(y_test_sample, y_pred_binary, average='binary')),
                "f1": float(f1_score(y_test_sample, y_pred_binary, average='binary'))
            }
            
            logger.info("Evaluation results:")
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.4f}")
            
            # Clean up
            del X_test_sample, y_test_sample, y_pred, y_pred_binary
            gc.collect()
            log_memory_usage()
            
            return float(1 - metrics["accuracy"]), len(self.X_test), metrics
            
        except Exception as e:
            log_exception(e)
            logger.error("Evaluation failed")
            return 0.0, len(self.X_test), {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

def load_data(client_id: int = 0):
    """Load and preprocess the data with different splits for each client."""
    logger.info(f"Loading and preprocessing data for client {client_id}")
    start_time = time.time()
    log_memory_usage()
    
    try:
        # Load data
        logger.debug("Reading CSV file")
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
        
        logger.debug("Creating train/test splits")
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
        
        # Clean up
        del data, X, y, X_temp, y_temp, X_val, y_val
        gc.collect()
        log_memory_usage()
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        log_exception(e)
        logger.error("Data loading failed")
        raise

def main():
    logger.info("Starting client")
    start_time = time.time()
    log_memory_usage()
    
    try:
        # Get client ID from environment variable or default to 0
        client_id = int(os.getenv('CLIENT_ID', '0'))
        logger.info(f"Client ID: {client_id}")
        
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
        
    except Exception as e:
        log_exception(e)
        logger.error("Client failed to complete")
        sys.exit(1)

if __name__ == "__main__":
    main() 