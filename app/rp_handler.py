import os
import sys

print(">>> rp_handler starting", flush=True)
sys.stdout.flush()

import runpod
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vton_inference import VTONRunner

device = "cuda" if torch.cuda.is_available() else "cpu"
runner = None

if device == "cuda":
    try:
        runner = VTONRunner(device=device)
    except Exception as init_err:
        print(f"Model init failed: {init_err}", flush=True)
        sys.exit(1)
else:
    print("No GPU detected, skipping model loading.", flush=True)


# Kevin Song +1
def handler(job):
    job_input = job.get("input", {})

    # Health-check bypass for RunPod cold-start probes
    job_id = str(job.get("id", ""))
    if (
            "local" in job_id
            or not job_input
            or (not job_input.get("user_image") and not job_input.get("garment_image"))
    ):
        return {"status": "success", "info": "Health check passed."}

    user_image = job_input.get("user_image")
    garment_image = job_input.get("garment_image")
    category = job_input.get("category", "tops")
    garment_photo_type = job_input.get("garment_photo_type", "flat-lay")
    num_timesteps = job_input.get("num_timesteps", 30)
    guidance_scale = job_input.get("guidance_scale", 1.5)
    seed = job_input.get("seed", 42)

    if not user_image or not garment_image:
        return {
            "status": "failed",
            "error": "Missing 'user_image' or 'garment_image' in request."
        }

    print(f"Job {job_id}: category={category}", flush=True)

    try:
        result_base64 = runner.predict(
            user_img_url=user_image,
            garment_img_url=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return {"status": "success", "image": result_base64}
    except Exception as run_err:
        print(f"Inference error: {run_err}", flush=True)
        return {
            "status": "failed",
            "error": f"Inference failed: {run_err}"
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})