FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Basic setup
RUN apt update && apt install -y \
    python3 python3-pip git \
    libgl1 libglib2.0-0 \
    build-essential \
    ninja-build \
    cmake \
    python3-dev

# Set working directory
COPY . /app
WORKDIR /app

# Pip setup
RUN pip3 install "pip<25.3"
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --upgrade pip setuptools wheel

# Install regular requirements
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir --upgrade diffusers transformers tokenizers

# --- BUILD XFORMERS FROM SOURCE (with submodules) ---
RUN git clone https://github.com/facebookresearch/xformers.git && \
    cd xformers && \
    git submodule update --init --recursive && \
    pip3 install -r requirements.txt && \
    pip3 install .

# --- END XFORMERS INSTALL ---

# Hugging Face token for download script
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Download models
RUN python3 download_models.py

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
