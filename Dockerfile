FROM ghcr.io/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# 保持原装官方源，等一下我们从外部注入你 Mac 本地的科学上网网络
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir --default-timeout=1000 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "numpy==1.26.4" \
    "runpod==1.9.1" \
    "diffusers==0.30.0" \
    "transformers==4.40.2" \
    "huggingface_hub==0.23.2" \
    "accelerate==0.29.3" \
    "einops" \
    "opencv-python-headless" \
    "Pillow" \
    "requests" \
    "tqdm"


COPY ./app /app
RUN mkdir -p /app/checkpoints

COPY test_input.json /test_input.json
COPY test_input.json /app/test_input.json

CMD ["python3", "-u", "/app/rp_handler.py"]
