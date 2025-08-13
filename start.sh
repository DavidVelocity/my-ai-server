#!/bin/bash
set -e

echo "Downloading models..."
python3 download_models.py

echo "Starting server..."
uvicorn main:app --host 0.0.0.0 --port 7860