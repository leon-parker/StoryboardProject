from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import os

# Ensure CUDA is available
assert torch.cuda.is_available(), "❌ CUDA is not available. Aborting."

# Load base SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "./models/sdxl",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Load SDXL Refiner pipeline
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Confirm base pipeline is fully on CUDA
for name, module in pipe.components.items():
    if hasattr(module, 'device'):
        assert module.device.type == "cuda", f"❌ {name} is on {module.device}, not CUDA!"

# Prompts
prompts = [
    "A child walks into a clean bathroom, realistic style, storyboard frame ",
    "The child picks up a colorful toothbrush, realistic lighting",
    "The child applies toothpaste onto the brush, clear and focused",
    "The child brushes their teeth in front of a mirror, morning scene",
    "The child spits and smiles in the mirror, happy and clean"
]

# Negative prompt
negative_prompt = "blurry, distorted, ugly, low quality, unrealistic, artefacts, distortion, blurry, scary, weird, no clutter, minimal background, simple cartoon style"

# Output folders
os.makedirs("Images/vanilla sdxl", exist_ok=True)
os.makedirs("Images/refined sdxl", exist_ok=True)

# Generate and refine images
for i, prompt in enumerate(prompts, start=1):
    print(f"🎨 Generating base image {i}: {prompt}")
    base_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    base_filename = f"Images/vanilla sdxl/image_{i}.png"
    base_image.save(base_filename)
    print(f"✅ Saved base image: {base_filename}")

    print(f"✨ Refining image {i}...")
    refined_image = refiner(
        prompt=prompt,
        image=base_image,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        strength=0.3
    ).images[0]

    refined_filename = f"Images/refined sdxl/image_{i}.png"
    refined_image.save(refined_filename)
    print(f"✅ Saved refined image: {refined_filename}")
