import runpod
import torch
from vton_inference import IDM_VTON_Runner

# 【只执行一次】容器启动时加载模型到显存，避免每次请求都重复加载（降低延迟）
device = "cuda" if torch.cuda.is_available() else "cpu"
runner = IDM_VTON_Runner(device=device)


def handler(job):
    """
    每次有用户点击试衣，这个函数就会被触发一次
    """
    # 1. 解析用户传过来的参数
    job_input = job.get('input', {})
    user_image = job_input.get('user_image')
    garment_image = job_input.get('garment_image')
    category = job_input.get('category', 'upper_body')

    if not user_image or not garment_image:
        return {"error": "缺少用户图片(user_image)或服装图片(garment_image)的URL"}

    try:
        # 2. 调用上面的推理核心
        result_base64 = runner.predict(user_image, garment_image, category)

        # 3. 返回给 RunPod API
        return {"status": "success", "image": result_base64}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


# 启动 RunPod 监听服务
runpod.serverless.start({"handler": handler})
