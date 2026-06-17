import os
import torch
import base64
import requests
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np

# 🔥【一致性修复】真正导入官方标准的 torchvision 语义分割网络组件
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionXLInpaintPipeline, DDIMScheduler
from transformers import (
    CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
    CLIPVisionModelWithProjection, CLIPImageProcessor
)


class IDM_VTON_Runner:
    def __init__(self, device="cuda"):
        self.device = device
        self.base_model_path = "/runpod-volume"

        print(f"正在从 {self.base_model_path} 初始模型组件...", flush=True)

        try:
            # ==================== 🔥 真正初始化 SOTA 级语义分割网络 ====================
            print("正在初始化 SOTA 级 DeepLabV3 人体语义分割网络...", flush=True)
            # 首次运行会自动从 PyTorch 官方下载约 160MB 的官方高精度预训练分割权重
            self.seg_weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.seg_model = deeplabv3_resnet50(weights=self.seg_weights).to(self.device)
            self.seg_model.eval()
            self.seg_transforms = self.seg_weights.transforms()
            # =========================================================================

            # 1. 加载核心 UNet
            self.unet = UNet2DConditionModel.from_pretrained(
                os.path.join(self.base_model_path, "unet"),
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
            )

            # 2. 加载变分自编码器 VAE
            self.vae = AutoencoderKL.from_pretrained(
                os.path.join(self.base_model_path, "vae"),
                torch_dtype=torch.float16,
            )

            # 3. 加载文本编码器与分词器
            self.text_encoder = CLIPTextModel.from_pretrained(
                os.path.join(self.base_model_path, "text_encoder"),
                torch_dtype=torch.float16,
            )

            self.tokenizer = CLIPTokenizer.from_pretrained(
                os.path.join(self.base_model_path, "tokenizer"),
            )

            try:
                self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    os.path.join(self.base_model_path, "text_encoder_2"),
                    torch_dtype=torch.float16,
                    use_safetensors=False,
                )
            except Exception:
                self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                    os.path.join(self.base_model_path, "text_encoder_2"),
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )

            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                os.path.join(self.base_model_path, "tokenizer_2"),
            )

            # 4. 加载 1280 维 CLIP 视觉编码器
            print("加载本地校准后的 1280维 CLIP 视觉编码器...", flush=True)
            clip_path = os.path.join(self.base_model_path, "ip-adapter", "image_encoder")
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                clip_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

            # 5. 加载图像预处理器
            if os.path.exists(os.path.join(clip_path, "preprocessor_config.json")):
                self.feature_extractor = CLIPImageProcessor.from_pretrained(clip_path)
            else:
                self.feature_extractor = CLIPImageProcessor()

            # 6. 加载调度器 Scheduler
            try:
                self.scheduler = DDIMScheduler.from_pretrained(
                    os.path.join(self.base_model_path, "scheduler"),
                )
            except Exception:
                self.scheduler = DDIMScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                )

            # 7. 组装 StableDiffusionXLInpaint 基础管道
            print("组装 Pipeline...", flush=True)
            self.pipeline = StableDiffusionXLInpaintPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                text_encoder_2=self.text_encoder_2,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                unet=self.unet,
                scheduler=self.scheduler,
                image_encoder=self.image_encoder,
                feature_extractor=self.feature_extractor,
            ).to(self.device)

            # 8. 加载 IP-Adapter 腾讯微调权重
            print("加载 IP-Adapter 权重并实施全闭环维度校准...", flush=True)
            ip_adapter_local = os.path.join(self.base_model_path, "ip-adapter")
            ip_adapter_weight = "ip-adapter_sdxl.bin"

            if os.path.exists(os.path.join(ip_adapter_local, ip_adapter_weight)):
                self.pipeline.load_ip_adapter(
                    pretrained_model_name_or_path_or_dict=ip_adapter_local,
                    subfolder=None,
                    weight_name=ip_adapter_weight,
                    image_encoder_folder=None,
                )
            else:
                print("本地未找到 IP-Adapter 权重, 从 HuggingFace 下载...", flush=True)
                self.pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name=ip_adapter_weight,
                    image_encoder_folder=None,
                )

            print("Pipeline 就绪。", flush=True)

        except Exception as e:
            print(f"模型加载失败: {str(e)}", flush=True)
            raise e

    def download_image(self, url):
        response = requests.get(url, timeout=15)
        return Image.open(BytesIO(response.content)).convert("RGB")

    def predict(self, user_img_url, garment_img_url, category="upper_body"):
        # 1. 极速拉取外网图片
        user_image = self.download_image(user_img_url)
        garment_image = self.download_image(garment_img_url)

        # 2. 严格对齐 IDM-VTON 的标准输入分辨率 768x1024
        user_image = user_image.resize((768, 1024))
        garment_image = garment_image.resize((768, 1024))

        # ==================== 3. 🔥【终极修复】SOTA 语义分割与维度严格对齐 ====================
        print("🚀 启动 DeepLabV3 语义像素分割并执行空间几何对齐...", flush=True)

        # 1. 图像过分割模型专用的标准化预处理
        input_tensor = self.seg_transforms(user_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            seg_output = self.seg_model(input_tensor)["out"]

        # 2. 提取全图概率最大的语义类别（COCO 标签中，15代表人类人像像素）
        seg_predictions = seg_output.argmax(1).squeeze(0).byte().cpu().numpy()

        # 3. 【核心救场修复】：由于网络内部缩放了尺寸，通过 Pillow 将语义小矩阵强行、精准放大回 768x1024 标准大物理画布
        seg_image_small = Image.fromarray(seg_predictions)
        seg_image_large = seg_image_small.resize((768, 1024), resample=Image.NEAREST)
        is_human_large = np.array(seg_image_large) == 15

        # 4. 初始化标准 1024x768 遮罩画布
        mask_np = np.zeros((1024, 768), dtype=np.uint8)

        # 5. 此时两侧矩阵的纵横维度达到了完美的 100% 绝对一致，执行安全的黄金躯干区域判定
        mask_np[220:850, :] = np.where(is_human_large[220:850, :], 255, 0)

        # 6. 应用大半径高斯模糊羽化（Radius=15）消除穿模白边
        final_mask_img = Image.fromarray(mask_np).convert("L")
        mask_image = final_mask_img.filter(ImageFilter.GaussianBlur(radius=15))
        # ===================================================================================

        # 4. 配置提示词 (根据 category 动态对齐衣服品类)
        garment_type = "garment" if category == "upper_body" else category
        prompt = (
            f"A photorealistic handsome male fitness model wearing the {garment_type} perfectly, "
            f"manly outfit, high quality, professional studio lighting, 8k resolution, detailed fabric texture"
        )
        negative_prompt = (
            "low quality, blurry, bra, sports bra, underwear, bikini, crop top, female garment, "
            "chest exposure, dress, woman clothes, distorted, deformed hands"
        )

        print("开始推理...", flush=True)

        # 5. 调用核心推理机制
        with torch.inference_mode():
            output_images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=user_image,
                mask_image=mask_image,
                ip_adapter_image=garment_image,
                num_inference_steps=45,
                guidance_scale=9.0,
            )

        # 6. 【防炸解析器】完美抠出生成的 PIL 图片对象
        print("🔮 推理迭代完成，正在进行最终生成的图像解析...", flush=True)
        final_image = None

        # 1. 核心修复：如果返回的对象拥有 .images 属性
        if hasattr(output_images, "images"):
            # 如果 images 是个列表且里面有图，直接剥离提取出第一张真正的 PIL 图像对象！
            if isinstance(output_images.images, list) and len(output_images.images) > 0:
                final_image = output_images.images[0]  # 👈 加上 [0]，彻底消灭 'list' object 报错
            else:
                final_image = output_images.images

        # 2. 备用防御：如果本身返回的就是个纯列表/元组
        elif isinstance(output_images, (tuple, list)) and len(output_images) > 0:
            first_item = output_images[0]
            # 如果列表里的第一个元素还是个列表，继续往下拆
            if isinstance(first_item, list) and len(first_item) > 0:
                final_image = first_item[0]
            else:
                final_image = first_item

        # 3. 备用防御：如果返回的直接就是一张 PIL 图像
        elif isinstance(output_images, Image.Image):
            final_image = output_images

        # 兜底安全性检查
        if final_image is None:
            raise RuntimeError(f"无法从推理返回对象中提取有效的图像！实际类型: {type(output_images)}")

        # 7. 转换 Base64 安全流输出返回
        buffered = BytesIO()
        final_image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("🎉【大功告成】试衣成功！图像已顺利转换为 Base64 编码流并安全交回接口！", flush=True)
        return img_str
