# realcartoon_optimized_storydiffusion.py
"""
RealCartoon XL v7 Optimized StoryDiffusion
Specifically tuned for RealCartoon XL v7 model characteristics:
- Cartoon/anime style prompting
- Optimal generation parameters for this model
- Character consistency tuned for cartoon aesthetics
- Enhanced negative prompts for cartoon quality
"""

import sys
import os
import torch
import numpy as np
import random
import copy
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

class RealCartoonOptimizedProcessor:
    """
    StoryDiffusion processor optimized for RealCartoon XL v7
    """
    def __init__(self, hidden_size=None, cross_attention_dim=None):
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.is_self_attention = cross_attention_dim is None
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global state
        
        # Reference mode: Store cartoon features
        if state.is_reference_mode and state.current_character and self.is_self_attention:
            step_key = f"{state.current_character}_{state.step_count}"
            if step_key not in state.reference_features:
                # Store high-quality cartoon features
                stored = hidden_states[0:1].clone().detach().to(torch.float32)  # Higher precision for cartoon details
                state.reference_features[step_key] = stored
        
        # Story mode: Apply cartoon consistency
        elif not state.is_reference_mode and state.current_character and self.is_self_attention:
            step_key = f"{state.current_character}_{state.step_count}"
            if step_key in state.reference_features:
                try:
                    stored = state.reference_features[step_key].to(hidden_states.device, dtype=hidden_states.dtype)
                    
                    if (stored.shape[1] <= hidden_states.shape[1] and 
                        stored.shape[2] == hidden_states.shape[2]):
                        
                        batch_size = hidden_states.shape[0]
                        if stored.shape[0] < batch_size:
                            stored = stored.repeat(batch_size, 1, 1)
                        
                        seq_len = hidden_states.shape[1]
                        if stored.shape[1] > seq_len:
                            stored = stored[:, :seq_len, :]
                        elif stored.shape[1] < seq_len:
                            padding_needed = seq_len - stored.shape[1]
                            last_token = stored[:, -1:, :].repeat(1, padding_needed, 1)
                            stored = torch.cat([stored, last_token], dim=1)
                        
                        # STRONG consistency for cartoon characters (they need to be very consistent)
                        strength = 0.85  # Higher for cartoon consistency
                        hidden_states = hidden_states * (1.0 - strength) + stored * strength
                        
                except Exception:
                    pass
        
        return self._standard_attention(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
    
    def _standard_attention(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        """Standard attention processing"""
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# Global state for RealCartoon optimization
class RealCartoonStoryState:
    def __init__(self):
        self.reference_features = {}
        self.current_character = None
        self.is_reference_mode = True
        self.step_count = 0

    def reset(self):
        self.step_count = 0

state = RealCartoonStoryState()

def setup_seed(seed):
    """Setup random seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_realcartoon_pipeline(model_path, device="cuda"):
    """
    Setup pipeline optimized for RealCartoon XL v7
    """
    print("🎨 REALCARTOON XL v7 OPTIMIZED STORYDIFFUSION")
    print("📄 Specifically tuned for cartoon/anime character consistency")
    print("🎯 Enhanced prompting and parameters for RealCartoon model")
    print("=" * 70)
    
    print(f"📁 Model: {os.path.basename(model_path)}")
    print(f"📱 Device: {device}")
    
    # Load RealCartoon XL v7
    print("📦 Loading RealCartoon XL v7 pipeline...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    # Optimize scheduler for RealCartoon (DPM++ works great with cartoon models)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
        solver_order=2
    )
    
    # Memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(1)
    
    # Try xFormers for better performance
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xFormers enabled for better performance")
    except:
        print("⚠️ xFormers not available - using standard attention")
    
    return pipe

def setup_realcartoon_storydiffusion(pipe):
    """
    Setup StoryDiffusion optimized for RealCartoon XL v7
    """
    print("🎭 Setting up RealCartoon-optimized StoryDiffusion...")
    
    processor_count = 0
    attn_procs = {}
    
    for name in pipe.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") 
            else pipe.unet.config.cross_attention_dim
        )
        
        # Apply to self-attention layers (focusing on up_blocks for cartoon consistency)
        if (cross_attention_dim is None and 
            ("up_blocks" in name or "mid_block" in name)):  # Include mid_block for better consistency
            
            attn_procs[name] = RealCartoonOptimizedProcessor(
                hidden_size=pipe.unet.config.block_out_channels[-1],
                cross_attention_dim=cross_attention_dim
            )
            processor_count += 1
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    
    pipe.unet.set_attn_processor(copy.deepcopy(attn_procs))
    print(f"✅ Applied RealCartoon-optimized processors to {processor_count} layers")
    return pipe

def get_realcartoon_prompts():
    """
    Get prompts specifically optimized for RealCartoon XL v7
    """
    # Character description optimized for RealCartoon
    character_desc = "a beautiful anime girl with long flowing brown hair, bright blue eyes, soft facial features, cute expression"
    
    # Reference prompts - consistent cartoon portrait style
    reference_prompts = [
        f"anime portrait of {character_desc}, facing forward, neutral expression, soft lighting, high quality anime art",
        f"anime portrait of {character_desc}, gentle smile, looking directly at viewer, beautiful anime style",
        f"anime portrait of {character_desc}, happy cheerful expression, detailed anime artwork, vibrant colors"
    ]
    
    # Story prompts - keeping cartoon portrait style but with outfit/expression variations
    story_prompts = [
        f"anime portrait of {character_desc}, wearing school uniform, student outfit, anime school girl style",
        f"anime portrait of {character_desc}, wearing casual summer dress, relaxed expression, summer anime vibes",
        f"anime portrait of {character_desc}, wearing winter coat and scarf, cozy winter anime aesthetic",
        f"anime portrait of {character_desc}, wearing elegant formal dress, sophisticated anime portrait",
        f"anime portrait of {character_desc}, wearing sporty athletic wear, energetic expression, sports anime style",
        f"anime portrait of {character_desc}, wearing cute pajamas, sleepy expression, nighttime anime mood"
    ]
    
    return character_desc, reference_prompts, story_prompts

def get_realcartoon_generation_params():
    """
    Generation parameters optimized for RealCartoon XL v7
    """
    return {
        # Reference generation (higher quality)
        "reference": {
            "num_inference_steps": 35,      # More steps for better cartoon quality
            "guidance_scale": 8.5,          # Higher guidance for cartoon style
            "height": 1024,                 # Full resolution
            "width": 1024,
            "negative_prompt": "photorealistic, realistic, 3d render, blurry, low quality, bad anatomy, bad hands, missing fingers, extra fingers, mutated hands, deformed, ugly, bad proportions, extra limbs, malformed limbs, bad eyes, cross-eyed, different eyes, inconsistent character, multiple people, different person, changing face, realistic skin texture"
        },
        # Story generation (optimized for consistency)
        "story": {
            "num_inference_steps": 30,      # Balanced quality/speed
            "guidance_scale": 7.5,          # Moderate guidance for consistency
            "height": 1024,
            "width": 1024,
            "negative_prompt": "photorealistic, realistic, 3d render, blurry, low quality, bad anatomy, bad hands, missing fingers, extra fingers, mutated hands, deformed, ugly, bad proportions, extra limbs, malformed limbs, bad eyes, cross-eyed, different eyes, inconsistent character, multiple people, different person, changing face, realistic skin texture, inconsistent art style"
        }
    }

def generate_realcartoon_references(pipe, character_desc, reference_prompts, seed=12345):
    """
    Generate reference images optimized for RealCartoon XL v7
    """
    global state
    
    print(f"\n🎬 REFERENCE GENERATION - RealCartoon XL v7 Style")
    state.is_reference_mode = True
    state.current_character = "main_character"
    
    reference_images = []
    params = get_realcartoon_generation_params()["reference"]
    
    for i, prompt in enumerate(reference_prompts):
        state.reset()
        
        # Setup seed for cartoon consistency
        ref_seed = seed + i * 111  # Spaced seeds for variety but consistency
        setup_seed(ref_seed)
        generator = torch.Generator(device=pipe.device).manual_seed(ref_seed)
        
        # Enhanced prompt for RealCartoon
        enhanced_prompt = f"masterpiece, best quality, {prompt}, beautiful anime art style, detailed cartoon illustration, vibrant colors, clean artwork"
        
        print(f"  🎨 Reference {i+1}: {prompt[:60]}...")
        
        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=params["negative_prompt"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            height=params["height"],
            width=params["width"],
            generator=generator,
        ).images[0]
        
        reference_images.append(image)
        print(f"  ✅ Generated RealCartoon reference {i+1}")
    
    print(f"✅ Generated {len(reference_images)} RealCartoon reference images")
    return reference_images

def generate_realcartoon_story(pipe, character_desc, story_prompts, seed=22345):
    """
    Generate story images with RealCartoon consistency
    """
    global state
    
    print(f"\n🎬 STORY GENERATION - RealCartoon Consistency Mode")
    state.is_reference_mode = False
    state.current_character = "main_character"
    
    story_images = []
    params = get_realcartoon_generation_params()["story"]
    
    for i, prompt in enumerate(story_prompts):
        state.reset()
        
        # Setup seed for story consistency (closer to reference seeds)
        story_seed = seed + i * 77  # Closer seeds for better consistency
        setup_seed(story_seed)
        generator = torch.Generator(device=pipe.device).manual_seed(story_seed)
        
        # Enhanced prompt with strong character consistency keywords
        specific_char = f"the exact same {character_desc}, identical face, same character, consistent features"
        enhanced_prompt = f"masterpiece, best quality, {prompt.replace(character_desc, specific_char)}, beautiful anime art style, detailed cartoon illustration, vibrant colors, clean artwork, character consistency"
        
        print(f"  🎨 Story {i+1}: {prompt[:60]}...")
        
        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=params["negative_prompt"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            height=params["height"],
            width=params["width"],
            generator=generator,
        ).images[0]
        
        story_images.append(image)
        print(f"  ✅ Generated RealCartoon story {i+1}")
    
    print(f"✅ Generated {len(story_images)} RealCartoon story images")
    return story_images

def create_realcartoon_grid(reference_images, story_images, title="RealCartoon XL v7 StoryDiffusion"):
    """
    Create an enhanced grid specifically for RealCartoon results
    """
    if not reference_images and not story_images:
        return None
    
    all_images = reference_images + story_images
    ref_count = len(reference_images)
    
    # Calculate grid layout
    total_images = len(all_images)
    cols = min(3, total_images)
    rows = (total_images + cols - 1) // cols
    
    # Image dimensions for grid
    img_size = 512
    margin = 15
    label_height = 50
    title_height = 80
    
    # Grid dimensions
    grid_width = cols * img_size + (cols + 1) * margin
    grid_height = rows * (img_size + label_height) + (rows + 1) * margin + title_height
    
    # Create grid with cartoon-friendly background
    grid = Image.new('RGB', (grid_width, grid_height), (245, 245, 250))  # Light purple-gray
    draw = ImageDraw.Draw(grid)
    
    # Try to load fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_label = ImageFont.truetype("arial.ttf", 18)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (grid_width - title_width) // 2
    draw.text((title_x, 25), title, fill=(70, 70, 70), font=font_title)
    
    # Add subtitle
    subtitle = "Character Consistency with RealCartoon XL v7"
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font_label)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (grid_width - subtitle_width) // 2
    draw.text((subtitle_x, 55), subtitle, fill=(100, 100, 100), font=font_label)
    
    # Place images with cartoon-style borders
    for i, img in enumerate(all_images):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = margin + col * (img_size + margin)
        y = title_height + margin + row * (img_size + label_height + margin)
        
        # Add cartoon-style border
        border_color = (120, 160, 255) if i < ref_count else (255, 120, 160)  # Blue for ref, pink for story
        draw.rectangle([x-3, y-3, x+img_size+3, y+img_size+3], fill=border_color)
        
        # Resize and paste image
        resized_img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid.paste(resized_img, (x, y))
        
        # Add label with cartoon styling
        if i < ref_count:
            label = f"📚 Reference {i + 1}"
            color = (20, 80, 180)
        else:
            story_idx = i - ref_count + 1
            label = f"🎭 Story {story_idx}"
            color = (180, 20, 80)
        
        label_y = y + img_size + 8
        draw.text((x, label_y), label, fill=color, font=font_label)
    
    return grid

def main():
    """
    Main function for RealCartoon XL v7 optimized StoryDiffusion
    """
    
    # Find RealCartoon model
    possible_paths = [
        "../models/realcartoonxl_v7.safetensors",
        "../../models/realcartoonxl_v7.safetensors",
        "../realcartoonxl_v7.safetensors",
        "models/realcartoonxl_v7.safetensors"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ RealCartoon XL v7 model not found!")
        print("📁 Looking for: realcartoonxl_v7.safetensors")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Setup RealCartoon pipeline
        pipe = setup_realcartoon_pipeline(model_path, device)
        
        # Setup StoryDiffusion for RealCartoon
        pipe = setup_realcartoon_storydiffusion(pipe)
        
        # Get RealCartoon-optimized prompts
        character_desc, reference_prompts, story_prompts = get_realcartoon_prompts()
        
        print(f"\n🎭 Character: {character_desc}")
        print(f"📚 References: {len(reference_prompts)} prompts")
        print(f"🎬 Story: {len(story_prompts)} prompts")
        
        # Generate with RealCartoon optimization
        reference_images = generate_realcartoon_references(
            pipe, character_desc, reference_prompts, seed=12345
        )
        
        story_images = generate_realcartoon_story(
            pipe, character_desc, story_prompts, seed=22345
        )
        
        # Save results
        output_dir = "realcartoon_storydiffusion_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual images
        for i, img in enumerate(reference_images):
            img.save(f"{output_dir}/realcartoon_ref_{i+1:02d}.png")
        
        for i, img in enumerate(story_images):
            img.save(f"{output_dir}/realcartoon_story_{i+1:02d}.png")
        
        # Create and save enhanced grid
        grid = create_realcartoon_grid(reference_images, story_images)
        if grid:
            grid_path = f"{output_dir}/realcartoon_complete_grid.png"
            grid.save(grid_path)
            print(f"🎨 RealCartoon grid saved: {grid_path}")
        
        print(f"\n🎉 REALCARTOON XL v7 STORYDIFFUSION COMPLETE!")
        print(f"✅ Generated {len(reference_images)} references + {len(story_images)} story images")
        print(f"📁 Saved to: {output_dir}/")
        print(f"🎨 Optimized for cartoon/anime character consistency")
        print(f"💪 Consistency strength: 85% (high for cartoon characters)")
        print(f"🎯 Perfect for RealCartoon XL v7 model!")
        
        # Show optimization details
        print(f"\n📊 RealCartoon Optimizations Applied:")
        print(f"  🎨 DPM++ Karras scheduler for better cartoon quality")
        print(f"  🔧 Higher consistency strength (0.85) for cartoon characters")
        print(f"  📝 Cartoon-specific prompting and negative prompts")
        print(f"  🎭 Mid-block attention for enhanced consistency")
        print(f"  🌈 Vibrant color optimization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()