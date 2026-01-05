#!/bin/bash
set -e

echo "Downloading models..."
python3 download_models.py

echo "Starting server..."
exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload