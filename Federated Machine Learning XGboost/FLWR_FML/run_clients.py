import subprocess
import time
import sys
import os
import logging
from datetime import datetime

# Set up logging
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_clients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_clients(num_clients: int = 3):
    """Run multiple client processes."""
    logger.info(f"Starting federated learning with {num_clients} clients")
    
    # Limit the number of clients to reduce memory pressure
    num_clients = min(num_clients, 5)  # Cap at 5 clients
    logger.info(f"Using {num_clients} clients to reduce memory pressure")
    
    processes = []
    
    # Start server
    logger.info("Starting server")
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(3)  # Wait for server to start
    
    # Start clients
    for i in range(num_clients):
        # Set environment variable for client ID
        env = os.environ.copy()
        env['CLIENT_ID'] = str(i)
        
        # Start client with environment variables
        logger.info(f"Starting client {i}")
        client_process = subprocess.Popen(
            [sys.executable, "client.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(client_process)
        time.sleep(2)  # Wait between starting clients to reduce memory pressure
    
    # Wait for all processes to complete
    try:
        logger.info("Waiting for server to complete")
        server_process.wait()
        
        for i, process in enumerate(processes):
            logger.info(f"Waiting for client {i} to complete")
            process.wait()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, terminating processes")
        server_process.terminate()
        for process in processes:
            process.terminate()
    
    logger.info("All processes completed")

if __name__ == "__main__":
    # Default to 3 clients, but can be overridden with command line argument
    num_clients = 3
    if len(sys.argv) > 1:
        try:
            num_clients = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid number of clients: {sys.argv[1]}, using default: 3")
    
    run_clients(num_clients=num_clients) 