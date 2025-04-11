import flwr as fl
import numpy as np
from typing import List, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import time
from collections import Counter
import psutil
import gc

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging to both file and console
log_filename = f"logs/server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Server memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

class XGBoostStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initializing XGBoostStrategy")
        self.round_start_time = None
        log_memory_usage()
    
    def aggregate_fit(
        self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, dict]:
        """Aggregate model updates from clients."""
        logger.debug(f"Round {rnd} - Starting aggregation")
        log_memory_usage()

        if not results:
            logger.error("No results received from clients")
            return None, {}
        
        # Log the number of successful and failed updates
        logger.info(f"Round {rnd} - Received {len(results)} successful updates")
        if failures:
            logger.error(f"Round {rnd} - Had {len(failures)} failures:")
            for failure in failures:
                logger.error(f"Failure: {str(failure)}")
        
        # Check parameter sizes
        sizes = [len(fit_res.parameters.tensors[0]) for _, fit_res in results]
        if len(set(sizes)) > 1:
            logger.error(f"Model size mismatch between clients: {sizes}")
            
            # Try to fix by using only clients with the most common size
            size_counts = Counter(sizes)
            most_common_size = size_counts.most_common(1)[0][0]
            
            # Filter results to only include clients with the most common size
            filtered_results = [
                (client, fit_res) for client, fit_res in results 
                if len(fit_res.parameters.tensors[0]) == most_common_size
            ]
            
            if not filtered_results:
                logger.error("No clients with consistent model size")
                return None, {}
                
            logger.info(f"Using {len(filtered_results)} clients with consistent model size: {most_common_size}")
            results = filtered_results
            
        logger.debug(f"All client models have consistent size: {sizes[0]} bytes")
        
        try:
            # Perform aggregation
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
            
            if aggregated_parameters is not None:
                logger.info(f"Round {rnd} aggregation successful")
                logger.debug(f"Aggregated model size: {len(aggregated_parameters.tensors[0])} bytes")
            else:
                logger.error(f"Round {rnd} aggregation failed")
            
            # Clean up
            gc.collect()
            log_memory_usage()
                
            return aggregated_parameters, aggregated_metrics
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            return None, {}

    def aggregate_evaluate(
        self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]], failures: List[BaseException],
    ) -> Tuple[float, dict]:
        """Aggregate evaluation metrics from clients."""
        logger.debug(f"Round {rnd} - Starting evaluation aggregation")
        logger.info(f"Round {rnd} - Number of evaluation results: {len(results)}")
        if failures:
            logger.error(f"Round {rnd} - Number of evaluation failures: {len(failures)}")
            for failure in failures:
                logger.error(f"Evaluation failure: {str(failure)}")
        
        log_memory_usage()
        
        if not results:
            return 0.0, {}

        try:
            # Initialize metrics
            total_examples = sum(r.num_examples for _, r in results)
            aggregated_metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
            
            # Aggregate metrics
            for _, r in results:
                weight = r.num_examples / total_examples
                for metric in aggregated_metrics.keys():
                    if metric in r.metrics:
                        aggregated_metrics[metric] += r.metrics[metric] * weight
            
            # Calculate aggregated loss
            loss_aggregated = sum(r.loss * r.num_examples for _, r in results) / total_examples
            
            logger.info(f"Round {rnd} evaluation finished successfully")
            logger.info(f"Loss: {loss_aggregated:.4f}")
            for metric, value in aggregated_metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.4f}")
            
            # Clean up
            gc.collect()
            log_memory_usage()
            
            return loss_aggregated, aggregated_metrics
        except Exception as e:
            logger.error(f"Error during evaluation aggregation: {str(e)}")
            return 0.0, {}

def main():
    logger.info("Starting Flower server")
    start_time = time.time()
    log_memory_usage()
    
    # Define the strategy with reduced minimum clients
    strategy = XGBoostStrategy(
        min_available_clients=1,  # Reduced from 2 to 1
        min_fit_clients=1,        # Reduced from 2 to 1
        min_evaluate_clients=1,   # Reduced from 2 to 1
        fraction_fit=1.0,         # Use all available clients for training
        fraction_evaluate=1.0,    # Use all available clients for evaluation
    )
    
    logger.info("Server configuration:")
    logger.info(f"Min available clients: {strategy.min_available_clients}")
    logger.info(f"Min fit clients: {strategy.min_fit_clients}")
    logger.info(f"Min evaluate clients: {strategy.min_evaluate_clients}")
    
    # Start Flower server with reduced rounds and timeout
    logger.info("Starting server on 127.0.0.1:8080")
    try:
        fl.server.start_server(
            server_address="127.0.0.1:8080",
            config=fl.server.ServerConfig(
                num_rounds=1,  # Reduced from 3 to 1 for testing
                round_timeout=60.0  # Add 60 second timeout per round
            ),
            strategy=strategy
        )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    
    total_duration = time.time() - start_time
    logger.info(f"Server finished after {total_duration:.2f} seconds")

if __name__ == "__main__":
    main() 