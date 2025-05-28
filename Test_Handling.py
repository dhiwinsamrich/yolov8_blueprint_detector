import os
import subprocess
import time
import socket
import unittest

# Configuration
APP_FILE = "app.py"  # Assuming your FastAPI app is in app.py
HOST = "127.0.0.1"

class TestPortHandling(unittest.TestCase):
    def _is_port_in_use(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((host, port))
                return True
            except socket.error:
                return False

    def _test_port(self, port_to_test):
        env = os.environ.copy()
        env["PORT"] = str(port_to_test)
        
        process = None
        try:
            # Start the FastAPI app with Uvicorn, using the PORT environment variable
            # Ensure uvicorn is installed: pip install uvicorn
            # Ensure your app is named 'app' within app.py, or adjust "app:app"
            command = ["uvicorn", f"{APP_FILE.replace('.py', '')}:app", "--host", HOST]
            
            print(f"\nAttempting to start server on port {port_to_test}...")
            # Uvicorn will pick up the PORT from the environment if --port is not specified
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Give the server a moment to start
            time.sleep(5) # Increased wait time for server to start

            # Check if the server process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"Server failed to start on port {port_to_test}.")
                print(f"STDOUT:\n{stdout.decode(errors='ignore')}")
                print(f"STDERR:\n{stderr.decode(errors='ignore')}")
                return False

            # Check if the port is in use
            if self._is_port_in_use(HOST, port_to_test):
                print(f"Server successfully started and listening on port {port_to_test}.")
                return True
            else:
                print(f"Server started but port {port_to_test} is not in use.")
                # Attempt to read output if server didn't bind to port correctly
                try:
                    stdout, stderr = process.communicate(timeout=2)
                    print(f"STDOUT:\n{stdout.decode(errors='ignore')}")
                    print(f"STDERR:\n{stderr.decode(errors='ignore')}")
                except subprocess.TimeoutExpired:
                    print("Process communication timed out.")
                return False
        
        finally:
            if process and process.poll() is None: # Check if process is running
                print(f"Terminating server on port {port_to_test}...")
                process.terminate()
                try:
                    process.wait(timeout=5) # Wait for termination
                except subprocess.TimeoutExpired:
                    print(f"Server on port {port_to_test} did not terminate gracefully, killing...")
                    process.kill()
                print(f"Server on port {port_to_test} terminated.")
            elif process:
                 print(f"Server process on port {port_to_test} already terminated with code {process.poll()}.")

    def test_default_port_uvicorn(self):
        print("\n--- Testing Uvicorn Default Port (8000 if PORT env var not set) ---")
        # Uvicorn's default is 8000 if --port is not specified and PORT env is not set
        # We are not setting PORT env var here to test this default
        env = os.environ.copy()
        if "PORT" in env: # Ensure PORT is not set for this specific test
            del env["PORT"]

        process = None
        try:
            command = ["uvicorn", f"{APP_FILE.replace('.py', '')}:app", "--host", HOST, "--port", "8000"]
            print(f"Attempting to start server on default Uvicorn port 8000...")
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)
            self.assertTrue(process.poll() is None, "Server failed to start on default port 8000")
            self.assertTrue(self._is_port_in_use(HOST, 8000), "Server not listening on default port 8000")
            print("Server successfully started on default port 8000.")
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait()

    def test_custom_port_5001(self):
        print("\n--- Testing Custom Port 5001 via PORT Environment Variable ---")
        self.assertTrue(self._test_port(5001), "Test failed for port 5001")

    def test_custom_port_5002(self):
        print("\n--- Testing Custom Port 5002 via PORT Environment Variable ---")
        self.assertTrue(self._test_port(5002), "Test failed for port 5002")

if __name__ == "__main__":
    # Note: Running these tests sequentially might sometimes cause issues with port binding
    # if a previous server instance doesn't shut down quickly enough.
    # Consider running tests individually if you encounter such problems.
    unittest.main()