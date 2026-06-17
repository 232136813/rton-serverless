import os
import base64
import requests
from io import BytesIO
from PIL import Image

from huggingface_hub import hf_hub_download
from fashn_vton import TryOnPipeline

# Backward-compatible category mapping: old API names -> FASHN VTON names
CATEGORY_MAP = {
    "upper_body": "tops",
    "lower_body": "bottoms",
    "full_body": "one-pieces",
    "tops": "tops",
    "bottoms": "bottoms",
    "one-pieces": "one-pieces",
}

# Kevin Song +1
def ensure_weights(weights_dir):
    """Download model weights from HuggingFace if not already present."""
    os.makedirs(weights_dir, exist_ok=True)

    # 1. TryOnModel weights
    model_path = os.path.join(weights_dir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"Downloading TryOnModel weights to {weights_dir} ...", flush=True)
        hf_hub_download(
            repo_id="fashn-ai/fashn-vton-1.5",
            filename="model.safetensors",
            local_dir=weights_dir,
        )

    # 2. DWPose ONNX models
    dwpose_dir = os.path.join(weights_dir, "dwpose")
    os.makedirs(dwpose_dir, exist_ok=True)

    for filename in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        if not os.path.exists(os.path.join(dwpose_dir, filename)):
            print(f"Downloading DWPose/{filename} ...", flush=True)
            hf_hub_download(
                repo_id="fashn-ai/DWPose",
                filename=filename,
                local_dir=dwpose_dir,
            )

    # 3. FashnHumanParser weights are auto-downloaded on first use
    print("Weights ready.", flush=True)

# Kevin Song +1
class VTONRunner:
    """FASHN VTON v1.5 inference wrapper for RunPod Serverless."""

    # Kevin Song
    def __init__(self, weights_dir="/runpod-volume/weights", device="cuda"):
        self.device = device
        self.weights_dir = weights_dir

        # Download weights if missing (first cold start on new volume)
        ensure_weights(weights_dir)

        print(f"Loading FASHN VTON v1.5 pipeline (device={device}) ...", flush=True)
        self.pipeline = TryOnPipeline(weights_dir=weights_dir, device=device)
        print("Pipeline ready.", flush=True)

    # Kevin Song
    @staticmethod
    def download_image(url):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    # Kevin Song +1
    def predict(
        self,
        user_img_url,
        garment_img_url,
        category="tops",
        garment_photo_type="flat-lay",
        num_timesteps=30,
        guidance_scale=1.5,
        seed=42,
    ):
        person_image = self.download_image(user_img_url)
        garment_image = self.download_image(garment_img_url)

        mapped_category = CATEGORY_MAP.get(category, "tops")

        print(
            f"Running inference: category={mapped_category}, "
            f"steps={num_timesteps}, seed={seed}",
            flush=True,
        )

        result = self.pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=mapped_category,
            garment_photo_type=garment_photo_type,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        buffered = BytesIO()
        result.images[0].save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("Inference complete.", flush=True)
        return img_str