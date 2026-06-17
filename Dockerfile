FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "fashn-vton @ git+https://github.com/fashn-AI/fashn-vton-1.5.git" \
    "runpod==1.9.1" \
    "requests"

COPY ./app /app

CMD ["python3", "-u", "/app/rp_handler.py"]