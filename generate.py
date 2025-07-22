from diffusers import StableDiffusionXLPipeline
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

# Confirm all components are on CUDA
for name, module in pipe.components.items():
    if hasattr(module, 'device'):
        assert module.device.type == "cuda", f"❌ {name} is on {module.device}, not CUDA!"

# Prompts
prompts = [
    "A child walks into a clean bathroom, realistic style, storyboard frame",
    "The child picks up a colorful toothbrush, realistic lighting",
    "The child applies toothpaste onto the brush, clear and focused",
    "The child brushes their teeth in front of a mirror, morning scene",
    "The child spits and smiles in the mirror, happy and clean"
]

# Negative prompt
negative_prompt = "blurry, distorted, ugly, low quality, unrealistic, artefacts, distortion, blurry, scary, weird, no clutter, minimal background, simple cartoon style"

# Output folder
output_dir = "Images/vanilla_sdxl"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
for i, prompt in enumerate(prompts, start=1):
    print(f"🎨 Generating image {i}: {prompt}")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,   # Reduced for speed
        guidance_scale=6.5        # Slightly lower for faster convergence
    ).images[0]

    filename = f"{output_dir}/image_{i}.png"
    image.save(filename)
    print(f"✅ Saved: {filename}")

print("🎉 All base images generated.")
