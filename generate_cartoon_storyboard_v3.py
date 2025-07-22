import subprocess
import sys
import importlib.util
import os
import torch
from PIL import Image

# === Auto-install required packages ===
def ensure_package(package):
    if importlib.util.find_spec(package) is None:
        print(f"⬇️ Installing missing package: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["diffusers", "transformers", "torch", "Pillow"]:
    ensure_package(pkg)

from diffusers import StableDiffusionXLPipeline

# === Paths ===
base_model_path = r"C:\Users\Leon Parker\Documents\Newcastle University\Semester 3\Individual Project\SDXL Diffusers\models\sdxl"
reference_image_path = r"C:\Users\Leon Parker\Documents\Newcastle University\Semester 3\Individual Project\SDXL Diffusers\assets\reference_character.png"
output_folder = "storyboard_frames_cartoon_v3"

# === Prompts: Focused on action, flat cartoon style ===
prompts = [
    "A cartoon child enters a clean white bathroom, wearing a red shirt and blue shorts, flat 2D cartoon style",
    "The same child picks up a red toothbrush and blue toothpaste at the sink",
    "The cartoon child squeezes blue toothpaste onto the red toothbrush, clean white bathroom, flat cartoon",
    "The child brushes their teeth in front of a mirror, foam on mouth, red shirt, flat cartoon, white background",
    "The child stands in front of a mirror with a sink, smiling, flat cartoon, only one child visible"

]

negative_prompt = (
    "realistic, photorealistic, 3D, shadow, cluttered, extra limbs, inconsistent face, different clothes, "
    "reflections, photographic, detailed textures, depth, darkness, cinematic, duplication, mess, artefact"
)

try:
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model path '{base_model_path}' does not exist.")
    if not os.path.exists(reference_image_path):
        raise FileNotFoundError(f"Reference image not found at '{reference_image_path}'")
    if not os.path.exists(os.path.join(base_model_path, "model_index.json")):
        raise FileNotFoundError("Missing 'model_index.json' in SDXL model folder.")

    generator = torch.Generator(device="cuda").manual_seed(1234)
    os.makedirs(output_folder, exist_ok=True)

    # === Step 1: Load Base Model ===
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # === Step 2: Load IP-Adapter ===
    ip_adapter_configs = [
        ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.safetensors"),
        ("h94/IP-Adapter", "sdxl_models", "ip-adapter-plus-face_sdxl_vit-h.safetensors"),
    ]

    ip_adapter_loaded = False
    for repo, subfolder, weight_name in ip_adapter_configs:
        try:
            print(f"🔄 Trying IP-Adapter: {weight_name}")
            pipe.load_ip_adapter(repo, subfolder=subfolder, weight_name=weight_name)
            pipe.set_ip_adapter_scale(0.6)  # 🔧 Reduced for better balance
            ip_adapter_loaded = True
            break
        except Exception as e:
            print(f"⚠️ Failed to load {weight_name}: {str(e)[:100]}...")

    if not ip_adapter_loaded:
        raise Exception("❌ No IP-Adapter loaded. Aborting.")

    # === Step 3: Load Reference Image ===
    reference_image = Image.open(reference_image_path).convert("RGB")

    # === Step 4: Generate Each Frame ===
    for i, prompt in enumerate(prompts, start=1):
        print(f"🖼️ Generating frame {i}: {prompt}")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=reference_image,
            num_inference_steps=40,
            guidance_scale=7.5,
            generator=generator
        )
        image = result.images[0]
        image.save(os.path.join(output_folder, f"frame_{i}.png"))

    print(f"\n✅ All storyboard frames saved to: '{output_folder}'")
    for i in range(1, len(prompts) + 1):
        print(f"   • Frame {i}: frame_{i}.png")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔍 Troubleshooting tips:")
    print("1. Ensure all model and adapter files are correct")
    print("2. Use strong action-specific prompts")
    print("3. Lower guidance_scale or adapter scale to reduce override")

finally:
    if 'pipe' in locals():
        del pipe
    torch.cuda.empty_cache()
