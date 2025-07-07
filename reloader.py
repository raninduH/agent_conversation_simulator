import os
import sys
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
# The command to run your application as a module
APP_COMMAND = [sys.executable, "-m", "agent_convo_simulator_app.main"]
# The directory to watch for changes (the current directory, since the reloader is inside the app folder)
WATCH_DIRECTORY = os.path.join(os.path.dirname(__file__), "agent_convo_simulator_app")
# --- End Configuration ---

class AppReloader(FileSystemEventHandler):
    """Restarts the application when a .py file is modified."""
    def __init__(self):
        self.process = None
        self.start_process()

    def start_process(self):
        """Start the application subprocess."""
        if self.process:
            print(">>> Terminating previous process...")
            self.process.terminate()
            self.process.wait()
        
        print(">>> Starting application...")
        self.process = subprocess.Popen(APP_COMMAND)

    def on_modified(self, event):
        """Handle file modification events."""
        # Ignore directory changes and non-Python files
        if event.is_directory:
            return
            
        # List of file extensions and patterns to ignore
        ignored_extensions = ['.json', '.db', '.sqlite', '.log', '.txt', '.md', '.pyc', '__pycache__']
        ignored_patterns = ['__pycache__', '.git', '.pytest_cache', 'node_modules']
        
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        
        # Check if file should be ignored
        should_ignore = False
        
        # Check extensions
        for ext in ignored_extensions:
            if file_path.lower().endswith(ext):
                should_ignore = True
                break
                
        # Check patterns
        for pattern in ignored_patterns:
            if pattern in file_path:
                should_ignore = True
                break
                
        # Only watch .py files
        if not file_path.endswith('.py'):
            should_ignore = True
            
        if should_ignore:
            print(f">>> Ignoring change in: {file_name}")
            return
            
        print(f">>> Change detected in {file_name}. Reloading...")
        print(f"    Full path: {file_path}")
        print(f"    Event type: {event.event_type}")
        self.start_process()

def main():
    event_handler = AppReloader()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=True)
    observer.start()
    
    print(f"--- Watching for changes in: {WATCH_DIRECTORY} ---")
    print("--- Press Ctrl+C to stop ---")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n>>> Stopping reloader...")
        observer.stop()
        if event_handler.process:
            event_handler.process.terminate()
    
    observer.join()

if __name__ == "__main__":
    main()
