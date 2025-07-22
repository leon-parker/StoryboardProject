from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterXL
from PIL import Image
import torch
import os

# ✅ Ensure CUDA is available
assert torch.cuda.is_available(), "❌ CUDA is not available. Aborting."

# ✅ Load base SDXL pipeline (NO variant="fp16")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "./models/sdxl",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# ✅ Load IP-Adapter for SDXL
ip_model_path = "./models/ip-adapter/ip-adapter_sdxl_vit-h.safetensors"
ip_config_path = "./models/ip-adapter/ip-adapter_sdxl_vit-h.json"
ip_adapter = IPAdapterXL(pipe, ip_model_path, ip_config_path, device="cuda")

# ✅ Load reference image
ref_image_path = "reference.png"
assert os.path.exists(ref_image_path), f"❌ Reference image not found at {ref_image_path}"
ref_image = Image.open(ref_image_path).convert("RGB")

# ✅ Shared character description and prompts
base_prompt = "A 7-year-old boy with short brown hair, wearing a blue T-shirt, flat cartoon style, clean background"
prompts = [
    f"{base_prompt}, walking into a clean bathroom",
    f"{base_prompt}, picking up a colorful toothbrush",
    f"{base_prompt}, applying toothpaste to the brush",
    f"{base_prompt}, brushing teeth in front of a mirror",
    f"{base_prompt}, smiling in the mirror after brushing"
]

# ✅ Negative prompt to reduce noise and artifacts
negative_prompt = "ugly, distorted, cluttered, messy, over-detailed, hyperrealistic, extra limbs, mutated, inconsistent, blurry"

# ✅ Output folder setup
output_dir = "Images/ipadapter_sdxl"
os.makedirs(output_dir, exist_ok=True)

# ✅ Generate and save consistent images
for i, prompt in enumerate(prompts, start=1):
    print(f"🎨 Generating image {i}: {prompt}")
    image = ip_adapter.generate(
        pil_image=ref_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=6.5,
        image_guidance_scale=1.5,
        seed=42
    )
    filename = f"{output_dir}/image_{i}.png"
    image.save(filename)
    print(f"✅ Saved: {filename}")

print("🎉 All consistent images generated with IP-Adapter-SDXL.")
