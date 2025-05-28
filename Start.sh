#!/bin/bash

# Default ports
BACKEND_PORT=8000
FRONTEND_PORT=8080

# Function to kill processes on a given port
kill_process_on_port() {
    PORT_TO_KILL=$1
    echo "Checking for existing process on port $PORT_TO_KILL..."
    # For Linux/macOS
    PID=$(lsof -t -i:$PORT_TO_KILL)
    # For Windows (Git Bash with netstat and taskkill)
    # PID=$(netstat -ano | grep LISTENING | grep :$PORT_TO_KILL | awk '{print $5}' | xargs -I {} tasklist /FI "PID eq {}" /NH | awk '{print $2}')

    if [ -n "$PID" ]; then
        echo "Process found on port $PORT_TO_KILL (PID: $PID). Attempting to kill..."
        kill -9 $PID
        # taskkill /F /PID $PID # For Windows
        echo "Process on port $PORT_TO_KILL killed."
    else
        echo "No process found on port $PORT_TO_KILL."
    fi
}

# Kill existing processes on the ports
kill_process_on_port $BACKEND_PORT
kill_process_on_port $FRONTEND_PORT

# Start the FastAPI backend (app.py)
echo "Starting FastAPI backend (app.py) on port $BACKEND_PORT..."
# Ensure uvicorn is installed (pip install uvicorn)
# Ensure fastapi is installed (pip install fastapi)
# Ensure ultralytics and other dependencies from requirements.txt are installed
(uvicorn app:app --host 0.0.0.0 --port $BACKEND_PORT) & # Run in background
BACKEND_PID=$!
echo "FastAPI backend started with PID $BACKEND_PID."

# Wait a few seconds for the backend to initialize
sleep 5

# Start a simple HTTP server for frontend.html
# This requires Python to be installed.
# The frontend.html will be accessible at http://localhost:8080/frontend.html
if [ -f "frontend.html" ]; then
    echo "Starting HTTP server for frontend.html on port $FRONTEND_PORT..."
    (python -m http.server $FRONTEND_PORT) & # Run in background
    FRONTEND_PID=$!
    echo "HTTP server for frontend started with PID $FRONTEND_PID."
    echo "Frontend should be accessible at http://localhost:$FRONTEND_PORT/frontend.html"
else
    echo "frontend.html not found. Skipping frontend server."
fi

echo ""
_GREEN='\033[0;32m'
_NC='\033[0m' # No Color
echo -e "${_GREEN}Blueprint Detection Application Started!${_NC}"
echo "-----------------------------------------"
echo "Backend API (FastAPI): http://localhost:$BACKEND_PORT"
echo "API Docs (Swagger):    http://localhost:$BACKEND_PORT/docs"
echo "API Docs (ReDoc):      http://localhost:$BACKEND_PORT/redoc"
if [ -f "frontend.html" ]; then
    echo "Frontend (HTML):       http://localhost:$FRONTEND_PORT/frontend.html"
fi
echo "-----------------------------------------"
echo "To stop the servers, you may need to manually kill the processes."
echo "Backend PID: $BACKEND_PID"
if [ -f "frontend.html" ]; then
    echo "Frontend PID: $FRONTEND_PID"
    echo "You can use 'kill $BACKEND_PID $FRONTEND_PID' or 'pkill -f uvicorn' and 'pkill -f http.server'"
else
    echo "You can use 'kill $BACKEND_PID' or 'pkill -f uvicorn'"
fi

# Keep script running to show PIDs, or use 'wait' if you want it to exit only when children exit
# For simplicity, this script will exit, but the background processes will continue.
# To make the script wait for background processes (optional):
# wait $BACKEND_PID
# if [ -f "frontend.html" ]; then
# wait $FRONTEND_PID
# fi