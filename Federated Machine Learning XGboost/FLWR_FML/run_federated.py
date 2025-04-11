import subprocess
import time
import os
import sys

def run_command(command, name):
    print(f"Starting {name}...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return process

def main():
    # Start the server
    server_process = run_command([sys.executable, "server.py"], "server")
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Start clients
    client_processes = []
    for i in range(3):
        # Set environment variable for client ID
        env = os.environ.copy()
        env["CLIENT_ID"] = str(i)
        
        # Start client
        client_process = subprocess.Popen(
            [sys.executable, "client.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        client_processes.append(client_process)
        print(f"Started client {i}")
        time.sleep(2)  # Wait between client starts
    
    # Wait for all processes to complete
    try:
        server_process.wait()
        for client_process in client_processes:
            client_process.wait()
    except KeyboardInterrupt:
        print("Stopping all processes...")
        server_process.terminate()
        for client_process in client_processes:
            client_process.terminate()
    
    print("All processes completed")

if __name__ == "__main__":
    main() 