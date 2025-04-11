import os
import shutil
import logging
from datetime import datetime
import time

def clear_logs():
    """Clear all log files from the logs directory."""
    logs_dir = "logs"
    
    # Check if logs directory exists
    if not os.path.exists(logs_dir):
        print(f"Logs directory '{logs_dir}' does not exist. Nothing to clear.")
        return
    
    # Count files before deletion
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    num_files = len(log_files)
    
    if num_files == 0:
        print("No log files found to clear.")
        return
    
    # Create a backup directory with timestamp
    backup_dir = f"logs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Move log files to backup directory
    moved_files = 0
    for log_file in log_files:
        src_path = os.path.join(logs_dir, log_file)
        dst_path = os.path.join(backup_dir, log_file)
        try:
            shutil.move(src_path, dst_path)
            moved_files += 1
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not move {log_file}: {e}")
            # Try to copy instead of move
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied {log_file} instead of moving it.")
                moved_files += 1
            except (PermissionError, OSError) as e:
                print(f"Error: Could not copy {log_file}: {e}")
    
    print(f"Cleared {moved_files} out of {num_files} log files.")
    print(f"Logs backed up to: {backup_dir}")

if __name__ == "__main__":
    clear_logs() 