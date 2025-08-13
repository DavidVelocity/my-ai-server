FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip git libgl1 libglib2.0-0 build-essential

WORKDIR /app
COPY . /app

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools wheel

# Pin pip if needed for older deps
RUN pip3 install "pip<25.3"

# Install torch + torchvision + torchaudio with matching CUDA
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install the rest of requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Upgrade core ML libs (but not torch)
RUN pip3 install --no-cache-dir --upgrade diffusers transformers tokenizers

# Try installing prebuilt xformers first; if it fails, build from source
RUN pip install xformers==0.0.27.post2 --extra-index-url https://download.pytorch.org/whl/cu121 || \
    (echo "Prebuilt xformers failed, building from source..." && \
     pip uninstall -y xformers && \
     pip install ninja && \
     pip install --no-build-isolation git+https://github.com/facebookresearch/xformers.git)

# Make sure start.sh is executable
RUN chmod +x /app/start.sh

# Start the app
CMD ["./start.sh"]
