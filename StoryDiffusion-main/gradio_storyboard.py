# Gradio plugin for RealCartoon‑XL storyboard generation with modern schedulers.
#
# Prerequisites (Python ≥3.9):
#   pip install diffusers==0.30.0 transformers accelerate safetensors gradio==4.28.0
#
# Environment variable RCX_MODEL_DIR can point to your RealCartoonXL_v7 folder.
# Otherwise place the checkpoint folder or .safetensors file in ./models/RealCartoonXL_v7.
#
# Optional LoRAs (drop into ./models/):
#   style_lora.safetensors   – any SDXL style‑LoRA (weight slider)
#   alex_lora.safetensors    – character LoRA for Alex
#   emma_lora.safetensors    – character LoRA for Emma
#
# Launch: python gradio_storyboard_plugin.py

import os
import torch
import gradio as gr
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
)
from diffusers import LoraLoaderMixin

MODEL_PATH = os.getenv("RCX_MODEL_DIR", "./models/RealCartoonXL_v7")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True,
)
pipe.to(device)
pipe.enable_vae_tiling()
pipe.enable_xformers_memory_efficient_attention()

# --- Optional LoRA adapters --------------------------------------------------
LORA_FILES = {
    "style": "./models/style_lora.safetensors",
    "alex": "./models/alex_lora.safetensors",
    "emma": "./models/emma_lora.safetensors",
}
for name, path in LORA_FILES.items():
    if os.path.isfile(path):
        pipe = LoraLoaderMixin.load_lora_weights(pipe, path, adapter_name=name)
    else:
        print(f"[WARN] LoRA '{name}' not found at {path}; skipping.")

# -----------------------------------------------------------------------------

def _set_scheduler(name: str):
    """Replace the pipeline's scheduler in‑place."""
    if name == "DPM++ 2M Karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
            euler_at_final=True,
            solver_order=2,
        )
    elif name == "UniPC":
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            solver_order=2,
        )
    elif name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unknown scheduler {name}")


def generate(
    prompt: str,
    negative_prompt: str,
    scheduler_name: str,
    steps: int,
    guidance: float,
    seed: int,
    width: int,
    height: int,
    style_weight: float,
    alex_weight: float,
    emma_weight: float,
):
    """Core generation wrapper used by Gradio."""
    _set_scheduler(scheduler_name)

    adapters, weights = [], []
    if style_weight > 0 and "style" in pipe.adapters:
        adapters.append("style")
        weights.append(style_weight)
    if alex_weight > 0 and "alex" in pipe.adapters:
        adapters.append("alex")
        weights.append(alex_weight)
    if emma_weight > 0 and "emma" in pipe.adapters:
        adapters.append("emma")
        weights.append(emma_weight)

    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)

    generator = (
        torch.Generator(device=device).manual_seed(int(seed)) if seed >= 0 else None
    )

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(height),
        width=int(width),
        generator=generator,
    ).images[0]

    return image


with gr.Blocks(title="RealCartoon‑XL Storyboard Generator") as demo:
    gr.Markdown(
        "### RealCartoon‑XL Storyboard Generator\n"
        "Generate consistent cartoon panels with modern schedulers and LoRA control."
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="Alex waves to Emma at the playground...",
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                lines=2,
                placeholder="text, watermark, logo...",
            )
            scheduler_name = gr.Dropdown(
                ["DPM++ 2M Karras", "UniPC", "DDIM"],
                label="Scheduler",
                value="DPM++ 2M Karras",
            )
            steps = gr.Slider(10, 50, value=22, step=1, label="Inference steps")
            guidance = gr.Slider(1, 15, value=5, step=0.5, label="CFG (guidance scale)")
            seed = gr.Number(value=1234, precision=0, label="Seed (-1 = random)")
            width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
            height = gr.Slider(512, 1536, value=1024, step=64, label="Height")
            with gr.Accordion("LoRA Weights"):
                style_weight = gr.Slider(0, 1, value=0.3, step=0.05, label="Style LoRA weight")
                alex_weight = gr.Slider(0, 1, value=0.8, step=0.05, label="Alex LoRA weight")
                emma_weight = gr.Slider(0, 1, value=0.8, step=0.05, label="Emma LoRA weight")

            run_btn = gr.Button("Generate")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Result", height=512)

    run_btn.click(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            scheduler_name,
            steps,
            guidance,
            seed,
            width,
            height,
            style_weight,
            alex_weight,
            emma_weight,
        ],
        outputs=[output_image],
        api_name="generate",
    )

if __name__ == "__main__":
    demo.launch()
