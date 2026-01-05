FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip git libgl1 libglib2.0-0 build-essential

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir -r requirements.txt

RUN chmod +x /app/start.sh

CMD ["./start.sh"]
