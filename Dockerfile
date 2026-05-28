# 1. 使用官方带显卡驱动的 PyTorch 基础镜像
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# 2. 安装图像处理需要的系统底层依赖
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# 3. 安装 RunPod SDK 和 IDM-VTON 需要的 Python 依赖包
RUN pip install --no-cache-dir runpod diffusers transformers accelerate einops opencv-python Pillow requests

# 4. 【修改点】将本地的 app 文件夹内的所有内容，复制到 Docker 镜像的工作目录中
# 这样两个 Python 文件就会直接平铺在 Docker 镜像的 /app 目录下
COPY ./app /app

# 5. 【修改点】因为文件已经被复制到了 /app 根目录，启动命令保持不变
CMD ["python3", "-u", "rp_handler.py"]
