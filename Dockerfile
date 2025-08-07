FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Basic setup
RUN apt update && apt install -y \
    python3 python3-pip git \
    libgl1 libglib2.0-0 \
    build-essential \
    ninja-build \
    cmake \
    python3-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY . /app

# Install pip and PyTorch
RUN pip3 install --upgrade "pip<25.3" setuptools wheel
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir --upgrade diffusers transformers tokenizers

# --- BUILD XFORMERS FROM SOURCE (with submodules) ---
RUN git clone https://github.com/facebookresearch/xformers.git /xformers && \
    cd /xformers && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    pip3 install .

# Hugging Face token for download script
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Download models (assumes download_models.py uses the token correctly)
RUN python3 download_models.py

# Expose port and start the server
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
