import os
import sys
print(">>> BOOT: rp_handler 已启动", flush=True)
print(">>> /runpod-volume 存在:", os.path.exists("/runpod-volume"), flush=True)
print(">>> 内容:", os.listdir("/runpod-volume") if os.path.exists("/runpod-volume") else "挂载不存在", flush=True)
sys.stdout.flush()

import runpod
import torch


# 💡 确保平铺在同级目录下的推理脚本能被 Python 进程顺利读取
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vton_inference import IDM_VTON_Runner

# =========================================================================
# 【只在容器初始化开机时执行 1 次】
# 显卡冷启动时把 30多GB 模型塞满显存。后续自动伸缩弹性调用时，该类将长驻，实现秒级接单。
# =========================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
runner = None

try:
    runner = IDM_VTON_Runner(device=device)
except Exception as init_err:
    print(f"💥 容器初始化模型致命错误: {str(init_err)}")
    sys.exit(1)


def handler(job):
    """
    每次有电商买家在前端点击“试穿”时，本函数就会被 RunPod 自动触发并接单一单
    job 包含了外网 POST 进来的 JSON 结构体
    """
    # 1. 提取安全的入参字典
    job_input = job.get('input', {})
    user_image = job_input.get('user_image')  # 用户真人图网络 URL
    garment_image = job_input.get('garment_image')  # 衣服平铺/模特图网络 URL
    category = job_input.get('category', 'upper_body')  # 品类

    # 2. 基础安全性校验，防范前端漏传引发空指针崩溃
    if not user_image or not garment_image:
        return {
            "status": "failed",
            "error": "Missing parameter. Please check 'user_image' and 'garment_image' in your JSON payload."
        }

    print(f"收到有效的试衣请求！任务流水号 Job ID: {job.get('id')}")

    try:
        # 3. 扔进显存，启动 13 通道扩散引擎渲染，获取 Base64 成片
        result_base64 = runner.predict(
            user_img_url=user_image,
            garment_img_url=garment_image,
            category=category
        )

        # 4. 将高价值的 Base64 数据返回给 RunPod 异步队列节点
        return {
            "status": "success",
            "image": result_base64
        }

    except Exception as run_err:
        print(f"❌ 运行时推理发生严重崩溃: {str(run_err)}")
        return {
            "status": "failed",
            "error": f"AI inference crash on GPU worker: {str(run_err)}"
        }


# =========================================================================
# 启动 RunPod 官方 Serverless 服务总线监听，挂起网络长连接队列
# =========================================================================
runpod.serverless.start({"handler": handler})
