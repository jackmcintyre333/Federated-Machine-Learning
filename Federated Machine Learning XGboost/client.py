"""xgboost_quickstart: A Flower / XGBoost app."""

"""xgboost_quickstart: A Flower / XGBoost app."""

import warnings
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
import logging
from flwr.common.context import Context
from sklearn.metrics import f1_score, roc_auc_score
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

from xgboost_quickstart.task import load_data, replace_keys

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

    def _local_boost(self, bst_input):
        # Update trees based on local training data
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        return bst_input

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
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

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

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
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Make predictions based on the global model
        y_pred = bst.predict(self.valid_dmatrix)
        y_true = self.valid_dmatrix.get_label()

        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(f1),
            "AUC": float(auc)}
        )


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
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