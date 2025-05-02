# ---------------------------------------------
# Federated Learning Server with Flower (flwr)
# ---------------------------------------------
# This script defines a Flower server that coordinates federated learning
# rounds across multiple clients. It uses the FedAvg strategy to aggregate
# model parameters and custom metric aggregation functions to evaluate performance.

import flwr as fl  # Flower federated learning framework
import logging  # Python's standard logging library to log info to file and terminal

# ---------------------------------------------
# Logging Configuration
# Logs are written both to console and a file ("server.log")
# ---------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Log all INFO level messages and above
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format includes timestamp and level
    handlers=[
        logging.FileHandler("server.log"),  # Output logs to a file
        logging.StreamHandler()  # Also print logs to the terminal
    ]
)

# ---------------------------------------------
# Metric Aggregation Functions
# These are used to combine metrics reported from clients after each round
# ---------------------------------------------
def weighted_average_eval(metrics_list):
    # metrics_list is a list of tuples (num_examples, metrics_dict)
    total = sum(num_examples for num_examples, _ in metrics_list)
    return {
        "accuracy": sum(num_examples * m["accuracy"] for num_examples, m in metrics_list) / total,
        "precision": sum(num_examples * m["precision"] for num_examples, m in metrics_list) / total,
        "recall": sum(num_examples * m["recall"] for num_examples, m in metrics_list) / total,
        "f1_score": sum(num_examples * m["f1_score"] for num_examples, m in metrics_list) / total,
    } if total > 0 else {  # Fallback if no data is present
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0
    }

# Similar weighted average for training metrics (e.g. loss)
def weighted_average_fit(metrics_list):
    total = sum(num_examples for num_examples, _ in metrics_list)
    return {
        "train_loss": sum(num_examples * m["train_loss"] for num_examples, m in metrics_list) / total
    } if total > 0 else {"train_loss": 0.0}

# ---------------------------------------------
# Federated Averaging Strategy
# The server uses FedAvg and custom metric aggregation functions
# ---------------------------------------------
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average_eval,  # Aggregates accuracy, precision, recall, F1
    fit_metrics_aggregation_fn=weighted_average_fit  # Aggregates training loss
)

# ---------------------------------------------
# Start Flower Server
# Listens for clients, coordinates training rounds
# ---------------------------------------------
if __name__ == "__main__":
    logging.info("Starting Flower server...")
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=5),  # Total number of federated learning rounds
        strategy=strategy  # Use FedAvg strategy with our custom aggregators
    )
    logging.info("Flower server stopped.")