"""xgboost_quickstart: A Flower / XGBoost app."""

import warnings
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
import numpy as np
import logging
from flwr.common.context import Context
import xgboost as xgb
from flwr.client import Client, ClientApp
from flwr.common.config import unflatten_dict
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)

from my_test import load_data, replace_keys, setup_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client_logs'),
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# Define Flower Client
class FlowerClient(Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        logger.info(f"Initialized FlowerClient with {num_train} training samples and {num_val} validation samples")

    def _local_boost(self, bst_input):
        # Update trees based on local training data
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())
        logger.info(f"Completed local boosting for {self.num_local_round} rounds")
        return bst_input

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        logger.info(f"Starting fit operation for global round {global_round}")
        
        if global_round == 1:
            # First round local training
            logger.info("Performing first round local training")
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])
            
            # Load global model into booster
            bst.load_model(global_model)
            logger.info("Loaded global model for continued training")
            
            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)
        logger.info(f"Completed fit operation for round {global_round}")

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        logger.info("Starting evaluation")
        partician_id = self.context.node_config["partition-id"]
        logger.info(f"Partition ID: {partician_id}")
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)
        logger.info(f"Partition ID: {partician_id} Loaded global model for evaluation")

        # Make predictions based on the global model
        y_pred = bst.predict(self.valid_dmatrix)
        y_true = self.valid_dmatrix.get_label()
        
        # Log the predictions and true labels for comparison
        logging.info("Predictions vs True Labels:")
        for pred, true in zip(y_pred, y_true):
            logging.info(f"Prediction: {pred}, True Label: {true}")
        
        # logging.info("Predictions:")
        # for val in y_pred:
        #     logging.info(val)
        # logging.info("True labels:")
        # for val in y_true:
        #     logging.info(val)
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)
        logging.info(f"Partition ID: {partician_id} - Mean (Accuracy): %.2f", np.mean(y_pred_binary == y_true))
        logging.info(f"Partition ID: {partician_id} - True Positives: %d", np.sum((y_pred_binary == 1) & (y_true == 1)))
        logging.info(f"Partition ID: {partician_id} - True Negatives: %d", np.sum((y_pred_binary == 0) & (y_true == 0)))
        logging.info(f"Partition ID: {partician_id} - False Positives: %d", np.sum((y_pred_binary == 1) & (y_true == 0)))
        logging.info(f"Partition ID: {partician_id} - False Negatives: %d", np.sum((y_pred_binary == 0) & (y_true == 1)))
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)

        # Log evaluation metrics
        metrics_log = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(f1),
            "AUC": float(auc)
        }
        
        logger.info("Evaluation metrics:")
        for metric_name, metric_value in metrics_log.items():
            logger.info(f"Partition ID: {partician_id} - {metric_name}: {metric_value:.4f}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics=metrics_log
        )


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    logger.info(f"Initializing client for partition {partition_id} of {num_partitions}")
    
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(
        partition_id, num_partitions
    )

    cfg = replace_keys(unflatten_dict(context.run_config))
    num_local_round = cfg.get("local_epochs", 1)  # Default to 1 if not provided

    # Default XGBoost parameters
    default_params = {
        'max_depth': 3, 
        'eta': 0.1,  
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'lambda': 1.0,
        'alpha': 0.1
    }

    # Use params from config if available, otherwise use default
    xgb_params = cfg.get('params', default_params)
    logger.info("XGBoost parameters:")
    logger.info(xgb_params)

    # Return Client instance
    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        xgb_params,
    )

# Flower ClientApp
app = ClientApp(
    client_fn,
)