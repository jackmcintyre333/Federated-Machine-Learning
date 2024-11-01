"""xgboost_quickstart: A Flower / XGBoost app."""

import warnings
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
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

from my_test import load_data, replace_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logfileHigs'),
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
        logger.info("Starting local boost operation")
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())
            if i % 5 == 0:  # Log every 5 rounds
                logger.info(f"Completed boost round {i}/{self.num_local_round}")

        # Bagging: extract the last N=num_local_round trees for server aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]
        logger.info("Completed local boost operation")
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        logger.info(f"Starting fit operation for global round {global_round}")
        
        if global_round == 1:
            logger.info("Performing first round local training")
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
            logger.info("Completed first round training")
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
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)
        logger.info("Loaded global model for evaluation")

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        
        logger.info("Evaluation metrics:")
        logger.info(f"AUC: {auc:.4f}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc},
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
    num_local_round = cfg["local_epochs"]
    
    logger.info("Configuration loaded:")
    logger.info(f"Number of local epochs: {num_local_round}")
    logger.info("XGBoost parameters:")
    logger.info(cfg["params"])

    # Return Client instance
    return FlowerClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        cfg["params"],
    )

# Flower ClientApp
app = ClientApp(
    client_fn,
)