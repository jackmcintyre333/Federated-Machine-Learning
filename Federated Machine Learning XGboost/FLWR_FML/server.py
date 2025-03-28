import flwr as fl
import numpy as np
from typing import List, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import time

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging to both file and console
log_filename = f"logs/server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XGBoostStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initializing XGBoostStrategy")
        self.round_start_time = None
    
    def aggregate_fit(
        self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, dict]:
        """Aggregate model updates from clients."""
        if self.round_start_time is None:
            self.round_start_time = time.time()
            logger.info(f"Starting round {rnd} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info(f"Round {rnd} - Waiting for client updates...")
        logger.info(f"Round {rnd} - Number of results: {len(results)}")
        logger.info(f"Round {rnd} - Number of failures: {len(failures)}")
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_parameters is not None:
            round_duration = time.time() - self.round_start_time
            logger.info(f"Round {rnd} aggregation finished successfully after {round_duration:.2f} seconds")
            self.round_start_time = None
        else:
            logger.warning(f"Round {rnd} aggregation failed")
            
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]], failures: List[BaseException],
    ) -> Tuple[float, dict]:
        """Aggregate evaluation metrics from clients."""
        logger.info(f"Round {rnd} - Starting evaluation aggregation")
        logger.info(f"Round {rnd} - Number of evaluation results: {len(results)}")
        logger.info(f"Round {rnd} - Number of evaluation failures: {len(failures)}")
        
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(rnd, results, failures)
        
        if loss_aggregated is not None:
            logger.info(f"Round {rnd} evaluation finished successfully")
            logger.info(f"Loss: {loss_aggregated:.4f}")
            logger.info(f"Accuracy: {metrics_aggregated.get('accuracy', 0):.4f}")
            logger.info(f"Precision: {metrics_aggregated.get('precision', 0):.4f}")
            logger.info(f"Recall: {metrics_aggregated.get('recall', 0):.4f}")
            logger.info(f"F1 Score: {metrics_aggregated.get('f1', 0):.4f}")
        else:
            logger.warning(f"Round {rnd} evaluation failed")
            
        return loss_aggregated, metrics_aggregated

def main():
    logger.info("Starting Flower server")
    start_time = time.time()
    
    # Define the strategy
    strategy = XGBoostStrategy(
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
    )
    
    logger.info("Server configuration:")
    logger.info(f"Min available clients: {strategy.min_available_clients}")
    logger.info(f"Min fit clients: {strategy.min_fit_clients}")
    logger.info(f"Min evaluate clients: {strategy.min_evaluate_clients}")
    
    # Start Flower server
    logger.info("Starting server on 127.0.0.1:8080")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
    
    total_duration = time.time() - start_time
    logger.info(f"Server finished after {total_duration:.2f} seconds")

if __name__ == "__main__":
    main() 