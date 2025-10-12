#!/bin/bash
# Helper script to launch the HDBSCAN OHLCV Streamlit UI

echo "Starting HDBSCAN OHLCV Explorer..."
echo "=================================="
echo ""
echo "The UI will open in your default browser."
echo "Press Ctrl+C to stop the server."
echo ""

streamlit run app.py --server.port 8501 --server.address localhost
