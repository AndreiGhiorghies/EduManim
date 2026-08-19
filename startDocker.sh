cleanup() {
    echo "Stopping services..."
    kill $API_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "Starting API on port 8000..."
uvicorn backend.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 3

echo "Starting Gradio interface on port 7860..."
python3 frontend/app.py &
FRONTEND_PID=$!

wait -n $API_PID $FRONTEND_PID