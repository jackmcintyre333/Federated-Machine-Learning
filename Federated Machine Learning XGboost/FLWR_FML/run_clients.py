import subprocess
import time
import sys
import os

def run_clients(num_clients: int = 3):
    """Run multiple client processes."""
    processes = []
    
    # Start server
    server_process = subprocess.Popen([sys.executable, "server.py"])
    time.sleep(2)  # Wait for server to start
    
    # Start clients
    for i in range(num_clients):
        # Set environment variable for client ID
        env = os.environ.copy()
        env['CLIENT_ID'] = str(i)
        
        # Start client with environment variables
        client_process = subprocess.Popen(
            [sys.executable, "client.py"],
            env=env
        )
        processes.append(client_process)
        time.sleep(1)  # Wait between starting clients
    
    # Wait for all processes to complete
    server_process.wait()
    for process in processes:
        process.wait()

if __name__ == "__main__":
    run_clients(num_clients=3) 