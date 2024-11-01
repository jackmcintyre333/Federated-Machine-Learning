"""xgboost_quickstart: A Flower / XGBoost app using the Fridge dataset from CSV with stratified client partitioning and data logging."""
#code using just one dataset
import logging
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def setup_logger(partition_id):
    """Set up a logger for the current partition."""
    logger = logging.getLogger(f"client_{partition_id}")
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(f'logfile_client_{partition_id}', mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger
#must modify this code to get back to just using fridge data
def load_and_preprocess_data(file_path, logger):
    """Load, preprocess, and log information about the Fridge dataset."""
    logger.info(f"Loading data from {file_path}...")
    fridge_data = pd.read_csv(file_path)

    logger.info("Data info:")
    logger.info(fridge_data.info())

    
    logger.info("Data description:")
    logger.info(fridge_data.describe().to_string())
    
    logger.info("First few rows of the data:")
    logger.info(fridge_data.head().to_string())
    
    logger.info("Dropping 'type' column...")
    fridge_data = fridge_data.drop('type', axis=1)
    
    logger.info("Encoding categorical variables...")
    for col in fridge_data.columns:
        if fridge_data[col].dtype == 'object':
            logger.info(f"Encoding column: {col}")
            labelEncoder = LabelEncoder()
            fridge_data[col] = labelEncoder.fit_transform(fridge_data[col])

    
    logger.info("Data info after preprocessing:")
    logger.info(fridge_data.info())

    
    logger.info("Data description after preprocessing:")
    logger.info(fridge_data.describe().to_string())

    
    return fridge_data

def split_data(data, test_size=0.2, random_state=42, logger=None):
    """Split the data into train and test sets."""
    X = data.drop('label', axis=1)
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    if logger:
        logger.info(f"Train set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def transform_to_dmatrix(X, y):
    """Transform dataset to DMatrix format for xgboost."""
    return xgb.DMatrix(X, label=y)

def load_data(partition_id, num_clients):
    """Load and prepare Fridge data for two clients."""
    logger = setup_logger(partition_id)
    
    fridge_file_path = r"C:\Users\jackm\OneDrive\Documents\Bitbucket-ids\Federated Machine Learning GitHub\Federated-Machine-Learning\Federated Machine Learning XGboost\Datasets\Train_Test_Network.csv"  
    
    data = load_and_preprocess_data(fridge_file_path, logger)
    
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(data, logger=logger)
    
    logger.info(f"Creating {num_clients} equal partitions...")
    # Ensure num_clients is 2
    assert num_clients == 2, "This implementation supports only 2 clients"
    
    # Split the training data into two equal parts
    split_index = len(X_train) // 2
    if partition_id == 0:
        X_partition = X_train.iloc[:split_index]
        y_partition = y_train.iloc[:split_index]
    else:
        X_partition = X_train.iloc[split_index:]
        y_partition = y_train.iloc[split_index:]
    
    logger.info(f"Partition {partition_id} shape: {X_partition.shape}")
    logger.info(f"Describing Partition {partition_id}: {X_partition.describe().to_string()}")
    logger.info("Transforming data to DMatrix format...")
    train_dmatrix = transform_to_dmatrix(X_partition, y_partition)
    test_dmatrix = transform_to_dmatrix(X_test, y_test)
    
    num_train = len(X_partition)
    num_test = len(X_test)
    
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