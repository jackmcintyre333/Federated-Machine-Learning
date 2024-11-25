"""xgboost_quickstart: A Flower / XGBoost app."""

from typing import Dict

from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging
import logging
# Configure logging
logging.basicConfig(
    filename='logfile',          
    level=logging.DEBUG,         
    format='%(asctime)s - %(levelname)s - %(message)s',  
    filemode='w'                  
)
def evaluate_metrics_aggregation(eval_metrics):
    """Return aggregated metrics for evaluation."""
    logging.info(f"Evaluation metrics: {eval_metrics}")
    #Evaluation metrics: [(5995, {'AUC': 1.0, 'precision': 1.0, 'accuracy': 1.0, 'recall': 1.0, 'F1': 1.0}), (5995, {'AUC': 0.9998382923673997, 'precision': 1.0, 'accuracy': 0.6702251876563803, 'recall': 0.36060802069857695, 'F1': 0.5300689327311623})]
    total_num = 0
    for count, _ in eval_metrics:
        total_num += count
    logging.info(f"Total number of samples: {total_num}")
    
    metrics_aggregated = {}
    for metric in ["accuracy", "precision", "recall", "F1", "AUC"]:
        logging.info(f"Aggregating metric: {metric}")
        metric_sum = 0
        for count, metric_values in eval_metrics:
            metric_sum += metric_values[metric] * count
        metrics_aggregated[metric] = metric_sum / total_num
    
    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)