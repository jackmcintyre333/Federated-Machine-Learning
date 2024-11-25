import logging
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class PartitionFilter(logging.Filter):
    """Custom filter to add partition ID to log records."""
    def __init__(self, partition_id):
        super().__init__()
        self.partition_id = partition_id
    
    def filter(self, record):
        record.partition_id = self.partition_id
        return True

def setup_logger(partition_id):
    """Set up a logger for the current partition."""
    logger = logging.getLogger(f"client_{partition_id}")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers = []
    
    file_handler = logging.FileHandler(f'logfile_client_{partition_id}', mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Add partition ID to the format string
    formatter = logging.Formatter('%(asctime)s - Partition[%(partition_id)s] - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the partition filter
    partition_filter = PartitionFilter(partition_id)
    file_handler.addFilter(partition_filter)
    
    logger.addHandler(file_handler)
    
    return logger

def load_and_preprocess_data(file_path, logger):
    """Load, preprocess, and log information about the data dataset."""
    logger.info(f"Loading data from {file_path}...")
    data_data = pd.read_csv(file_path)
    
    logger.info("First few rows of the data:")
    logger.info(data_data.head().to_string())
    
    logger.info("Dropping 'type' column...")
    data_data = data_data.drop('type', axis=1)
    
    logger.info("Encoding categorical variables...")
    for col in data_data.columns:
        if data_data[col].dtype == 'object':
            logger.info(f"Encoding column: {col}")
            labelEncoder = LabelEncoder()
            data_data[col] = labelEncoder.fit_transform(data_data[col])
    
    logger.info(f"Data info after preprocessing: {data_data.head().to_string()}")
    
    return data_data

def split_data(data, test_size=0.2, random_state=42, logger=None):
    """Split the data into train and test sets."""
    X = data.drop('label', axis=1)
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    if logger:
        logger.info(f"Train set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Train set class distribution - Positive: {np.sum(y_train)}, Negative: {len(y_train) - np.sum(y_train)}")
        logger.info(f"Test set class distribution - Positive: {np.sum(y_test)}, Negative: {len(y_test) - np.sum(y_test)}")
    
    return X_train, X_test, y_train, y_test

def transform_to_dmatrix(X, y):
    """Transform dataset to DMatrix format for xgboost."""
    return xgb.DMatrix(X, label=y)

def load_data(partition_id, num_clients):
    """Load and prepare data data for clients."""
    logger = setup_logger(partition_id)
    
    data_file_path = r"C:\Users\jackm\OneDrive\Documents\Bitbucket-ids\Federated Machine Learning GitHub\Federated-Machine-Learning\Federated Machine Learning XGboost\Datasets\Train_Test_IoT_Fridge.csv"  
    
    logger.info(f"Starting data loading process for partition {partition_id}")
    data = load_and_preprocess_data(data_file_path, logger)
    logger.info(f"Data size: {data.shape}")
    # Split the training data into two equal parts
    split_index = len(data) // 2
    if partition_id == 0:
        data_partition = data[:split_index]
        logger.info(f"Using first half of data for partition {partition_id}")
    else:
        data_partition = data[split_index:]
        logger.info(f"Using second half of data for partition {partition_id}")
    
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(data_partition, logger=logger)
    
    # Detailed logging for the partition
    logger.info(f"Partition {partition_id} dataset details:")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    logger.info(f"Training set - Positive samples: {np.sum(y_train)}, Negative samples: {len(y_train) - np.sum(y_train)}")
    logger.info(f"Test set - Positive samples: {np.sum(y_test)}, Negative samples: {len(y_test) - np.sum(y_test)}")
    logger.info(f"Training set statistics:\n{X_train.describe().to_string()}")
    logger.info(f"Test set statistics:\n{X_test.describe().to_string()}")
    
    # Ensure num_clients is 2
    assert num_clients == 2, "This implementation supports only 2 clients"
    
    logger.info("Transforming data to DMatrix format...")
    train_dmatrix = transform_to_dmatrix(X_train, y_train)
    test_dmatrix = transform_to_dmatrix(X_test, y_test)
    
    num_train = len(X_train)
    num_test = len(X_test)
    
    logger.info(f"Completed data preparation for partition {partition_id}")
    
    return train_dmatrix, test_dmatrix, num_train, num_test

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict