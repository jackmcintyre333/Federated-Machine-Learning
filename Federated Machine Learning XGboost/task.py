import logging
from logging import INFO

import xgboost as xgb
from flwr.common import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Configure logging
logging.basicConfig(
    filename='logfileHigs',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

def train_test_split(partition, test_fraction, seed):
    """Split the data into train and validation set given split rate."""
    logger.info(f"Starting train-test split with test fraction {test_fraction} and seed {seed}")
    
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)
    
    logger.info(f"Split complete. Train samples: {num_train}, Test samples: {num_test}")
    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data):
    """Transform dataset to DMatrix format for xgboost."""
    logger.debug("Starting dataset to DMatrix transformation")
    x = data["inputs"]
    y = data["label"]
    logger.debug(f"Data shape - X: {x.shape}, y: {y.shape}")
    
    new_data = xgb.DMatrix(x, label=y)
    logger.debug("DMatrix transformation complete")
    return new_data


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_clients):
    """Load partition HIGGS data."""
    logger.info(f"Loading data for partition {partition_id} of {num_clients} clients")
    
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        logger.info("Initializing FederatedDataset")
        partitioner = IidPartitioner(num_partitions=num_clients)
        fds = FederatedDataset(
            dataset="jxie/higgs",
            partitioners={"train": partitioner},
        )
        logger.info("FederatedDataset initialization complete")

    # Load the partition for this `partition_id`
    logger.info(f"Loading partition {partition_id}")
    partition = fds.load_partition(partition_id, split="train")
    partition.set_format("numpy")
    logger.info("Partition loaded and format set to numpy")

    # Log the head of the dataset
    logger.info("Logging the head of the dataset before processing")
    dataset_head = partition[:5]
    logger.debug(f"Dataset head: {dataset_head}")

    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=42
    )

    # Reformat data to DMatrix for xgboost
    log(INFO, "Reformatting data...")
    logger.info("Starting DMatrix transformation")
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)
    logger.info("DMatrix transformation complete")

    logger.info(f"Data loading complete. Train samples: {num_train}, Validation samples: {num_val}")
    return train_dmatrix, valid_dmatrix, num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    logger.debug(f"Replacing '{match}' with '{target}' in dictionary keys")
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    logger.debug("Key replacement complete")
    return new_dict
