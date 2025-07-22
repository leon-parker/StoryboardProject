from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import os

# === Paths ===
base_model_path = "./models/sdxl"
reference_image_path = "./reference.jpg"
output_path = "./output_with_adapter.png"

# === Load SDXL Pipeline ===
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")

# ✅ Load IP-Adapter Plus from Hugging Face (includes CLIP)
pipe.load_ip_adapter(
    "h94/IP-Adapter",                      # repo name
    "sdxl_models",                         # folder with the model
    "ip-adapter-plus_sdxl_vit-h.safetensors"
)

# === Load reference image ===
if not os.path.exists(reference_image_path):
    raise FileNotFoundError(f"❌ Missing reference image: {reference_image_path}")

reference_image = Image.open(reference_image_path).convert("RGB")

# === Prompt ===
prompt = "A child brushes their teeth in front of a mirror, morning scene, photorealistic"

# === Generate image ===
print("🧠 Generating image with IP-Adapter...")
image = pipe(prompt=prompt, ip_adapter_image=reference_image).images[0]

# === Save result ===
image.save(output_path)
print(f"✅ Image saved to {output_path}")
