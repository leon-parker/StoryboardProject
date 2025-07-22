import torch
from PIL import Image
from torchvision import transforms
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

class IPAdapterXL:
    def __init__(self, pipe, device="cuda"):
        assert isinstance(pipe, StableDiffusionXLPipeline), "Pipeline must be an SDXL pipeline"

        self.pipe = pipe
        self.device = device

        # Model details
        ip_adapter_path = "."  # Root directory
        subfolder = "models/ip-adapter"
        weight_name = "ip-adapter-plus_sdxl_vit-h.safetensors"

        # ✅ Correct arguments for current diffusers version
        self.pipe.load_ip_adapter(
            pretrained_model_name_or_path=ip_adapter_path,
            subfolder=subfolder,
            weight_name=weight_name
        )

        self.pipe.set_ip_adapter_scale(1.0)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def generate(self, pil_image, prompt, negative_prompt=None,
                 num_inference_steps=30, guidance_scale=7.5,
                 image_guidance_scale=1.5, seed=None):

        # Convert reference image to tensor
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Optional seed
        generator = torch.manual_seed(seed) if seed is not None else None

        # Set IP-Adapter influence scale
        self.pipe.set_ip_adapter_scale(image_guidance_scale)

        # Generate image
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=image_tensor,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        return output.images[0]
