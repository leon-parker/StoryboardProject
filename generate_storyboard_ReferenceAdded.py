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

from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

# === Paths ===
base_model_path = r"C:\Users\Leon Parker\Documents\Newcastle University\Semester 3\Individual Project\SDXL Diffusers\models\sdxl"
output_folder = "storyboard_images_combined"

# === Prompts (visually clear + action-oriented) ===
prompts = [
    "A young child enters a brightly lit bathroom, realistic DSLR photo, consistent character",
    "The same child picks up a colorful toothbrush and tube of toothpaste, close-up hands, realistic photo",
    "The same child squeezes toothpaste onto the brush over the sink, foam visible, clean background",
    "The same child brushes their teeth in front of a mirror, foam on mouth, looking in mirror, realistic lighting",
    "The same child spits into the sink and smiles in the mirror, feeling clean, photo taken from behind"
]

# Negative prompt to reduce drift
negative_prompt = "inconsistent character, different person, multiple people, face change, inconsistent style"

try:
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Model path '{base_model_path}' does not exist.")
    if not os.path.exists(os.path.join(base_model_path, "model_index.json")):
        raise FileNotFoundError("Missing 'model_index.json' in SDXL model folder.")

    generator = torch.Generator(device="cuda").manual_seed(42)
    os.makedirs(output_folder, exist_ok=True)

    # === Method 1: IP-Adapter Approach ===
    try:
        print("🔁 Attempting IP-Adapter method...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        print(f"🖼️ Generating frame 1: {prompts[0]}")
        first_result = pipe(
            prompt=prompts[0],
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        )
        first_image = first_result.images[0]
        first_image.save(os.path.join(output_folder, "frame_1.png"))

        ip_adapter_configs = [
            ("h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl.safetensors"),
            ("h94/IP-Adapter", "sdxl_models", "ip-adapter-plus-face_sdxl_vit-h.safetensors"),
        ]

        ip_adapter_loaded = False
        for repo, subfolder, weight_name in ip_adapter_configs:
            try:
                print(f"🔄 Trying IP-Adapter: {weight_name}")
                pipe.load_ip_adapter(repo, subfolder=subfolder, weight_name=weight_name)
                pipe.set_ip_adapter_scale(0.75)  # Stronger character locking
                ip_adapter_loaded = True
                break
            except Exception as e:
                print(f"⚠️ Failed to load {weight_name}: {str(e)[:100]}...")
                continue

        if ip_adapter_loaded:
            reference_image = first_image  # lock in character from frame 1
            for i, prompt in enumerate(prompts[1:], start=2):
                print(f"🖼️ Generating frame {i}: {prompt}")
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=reference_image,
                    num_inference_steps=30,
                    guidance_scale=6.5,  # Slightly lower for prompt balance
                    generator=generator
                )
                image = result.images[0]
                image.save(os.path.join(output_folder, f"frame_{i}.png"))
        else:
            raise Exception("No compatible IP-Adapter found")

    except Exception as ip_error:
        print(f"⚠️ IP-Adapter method failed: {str(ip_error)[:100]}...")
        print("🔄 Falling back to Img2Img method...")

        if 'pipe' in locals():
            del pipe
        torch.cuda.empty_cache()

        # === Method 2: Img2Img fallback ===
        txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        print(f"🖼️ Generating frame 1 (Img2Img method): {prompts[0]}")
        first_result = txt2img_pipe(
            prompt=prompts[0],
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.0,
            generator=generator
        )
        first_image = first_result.images[0]
        first_image.save(os.path.join(output_folder, "frame_1.png"))

        del txt2img_pipe
        torch.cuda.empty_cache()

        img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        current_image = first_image
        for i, prompt in enumerate(prompts[1:], start=2):
            print(f"🖼️ Generating frame {i} (Img2Img): {prompt}")
            result = img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=current_image,
                strength=0.25,  # Stronger image retention
                num_inference_steps=30,
                guidance_scale=7.0,
                generator=generator
            )
            image = result.images[0]
            image.save(os.path.join(output_folder, f"frame_{i}.png"))
            current_image = image

        pipe = img2img_pipe

    print(f"\n✅ All storyboard frames saved to: '{output_folder}'")
    print("📋 Generated frames:")
    for i in range(1, len(prompts) + 1):
        print(f"   • Frame {i}: frame_{i}.png")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔍 Troubleshooting suggestions:")
    print("1. Check diffusers version: pip install --upgrade diffusers")
    print("2. Verify SDXL model structure and files")
    print("3. Try running with more GPU memory available")
    print("4. Consider using CPU mode: change .to('cuda') to .to('cpu')")

finally:
    if 'pipe' in locals():
        del pipe
    if 'txt2img_pipe' in locals():
        del txt2img_pipe
    if 'img2img_pipe' in locals():
        del img2img_pipe
    torch.cuda.empty_cache()
