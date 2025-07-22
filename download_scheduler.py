import logging
logging.basicConfig(level=logging.DEBUG)

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd3.5-medium",
    safety_checker=None,
    torch_dtype=torch.float32,
    variant=None,
    use_safetensors=True
).to("cuda")

prompt = "a consistent character sitting in a classroom, educational style"
image = pipe(prompt).images[0]
image.save("output.png")
print("✅ Image saved as output.png")