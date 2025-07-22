# fixed_story_prompts.py
"""
Fixed StoryDiffusion with Better Prompt Strategy
The key is using CONSISTENT prompting that matches the reference style
"""

import sys
import os
import torch
import numpy as np
import random
import copy
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# Use the working hot-plug processor from before
class TrulyHotPlugProcessor:
    """Same processor as before - this part works"""
    def __init__(self, hidden_size=None, cross_attention_dim=None):
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.is_self_attention = cross_attention_dim is None
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global state
        
        # Reference mode: Store features
        if state.is_reference_mode and state.current_character and self.is_self_attention:
            step_key = f"{state.current_character}_{state.step_count}"
            if step_key not in state.reference_features:
                stored = hidden_states[0:1].clone().detach()
                state.reference_features[step_key] = stored
        
        # Story mode: Apply consistency (with better strength)
        elif not state.is_reference_mode and state.current_character and self.is_self_attention:
            step_key = f"{state.current_character}_{state.step_count}"
            if step_key in state.reference_features:
                try:
                    stored = state.reference_features[step_key]
                    
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
                        
                        # STRONGER consistency for story mode
                        strength = 0.8  # Increased from 0.6
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

# Global state
class SimpleStoryState:
    def __init__(self):
        self.reference_features = {}
        self.current_character = None
        self.is_reference_mode = True
        self.step_count = 0

    def reset(self):
        self.step_count = 0

state = SimpleStoryState()

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_hotplug_storydiffusion(pipe):
    """Hot-plug StoryDiffusion"""
    print("🔌 HOT-PLUGGING StoryDiffusion")
    
    processor_count = 0
    attn_procs = {}
    
    for name in pipe.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") 
            else pipe.unet.config.cross_attention_dim
        )
        
        # Apply to self-attention in up_blocks
        if (cross_attention_dim is None and "up_blocks" in name):
            attn_procs[name] = TrulyHotPlugProcessor(
                hidden_size=pipe.unet.config.block_out_channels[-1],
                cross_attention_dim=cross_attention_dim
            )
            processor_count += 1
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    
    pipe.unet.set_attn_processor(copy.deepcopy(attn_procs))
    print(f"✅ Hot-plugged {processor_count} processors")
    return pipe

def generate_with_fixed_strategy(pipe, character_name, character_desc):
    """
    FIXED: Generate with better prompt strategy for consistency
    """
    global state
    
    print("🎯 FIXED PROMPT STRATEGY FOR STORYDIFFUSION")
    print("The key: Keep prompts SIMPLE and CONSISTENT!")
    
    # BETTER REFERENCE PROMPTS - Simple, consistent style
    reference_prompts = [
        f"portrait of {character_desc}, facing forward, neutral expression",
        f"portrait of {character_desc}, slight smile, looking at camera", 
        f"portrait of {character_desc}, happy expression, clear face"
    ]
    
    # BETTER STORY PROMPTS - Keep same style/framing as references
    story_prompts = [
        f"portrait of {character_desc}, wearing office clothes, professional look",
        f"portrait of {character_desc}, wearing casual clothes, relaxed expression",
        f"portrait of {character_desc}, wearing sports clothes, energetic smile",
        f"portrait of {character_desc}, wearing formal clothes, confident look",
        f"portrait of {character_desc}, wearing winter clothes, warm smile"
    ]
    
    print("💡 Key changes:")
    print("  • References: Simple portraits, consistent framing")
    print("  • Story: SAME framing (portraits), only clothes/expression change")
    print("  • No complex scenes that break feature consistency!")
    
    # Generate references
    print(f"\n🎬 REFERENCE MODE: {character_name}")
    state.is_reference_mode = True
    state.current_character = character_name
    
    reference_images = []
    base_seed = 12345
    
    for i, prompt in enumerate(reference_prompts):
        state.reset()
        setup_seed(base_seed + i * 100)
        generator = torch.Generator(device=pipe.device).manual_seed(base_seed + i * 100)
        
        enhanced_prompt = f"masterpiece, best quality, highly detailed, {prompt}, anime style, consistent character"
        negative_prompt = "bad anatomy, bad hands, missing fingers, blurry, low quality, different person, inconsistent character, multiple people"
        
        print(f"  🎨 Reference {i+1}: {prompt[:60]}...")
        
        image = pipe(
            enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=768,
            width=768,
            generator=generator,
        ).images[0]
        
        reference_images.append(image)
        print(f"  ✅ Generated reference {i+1}")
    
    # Generate story with FIXED approach
    print(f"\n🎬 STORY MODE: {character_name}")
    state.is_reference_mode = False
    state.current_character = character_name
    
    story_images = []
    
    for i, prompt in enumerate(story_prompts):
        state.reset()
        # Use VERY similar seeds for consistency
        setup_seed(base_seed + 50 + i * 25)  # Closer seeds!
        generator = torch.Generator(device=pipe.device).manual_seed(base_seed + 50 + i * 25)
        
        # CRITICAL: Use more specific character description
        specific_desc = f"the exact same {character_desc}, identical facial features, same face"
        enhanced_prompt = f"masterpiece, best quality, highly detailed, {prompt.replace(character_desc, specific_desc)}, anime style, consistent character"
        negative_prompt = "bad anatomy, bad hands, missing fingers, blurry, low quality, different person, inconsistent character, multiple people, changing face"
        
        print(f"  🎨 Story {i+1}: {prompt[:60]}...")
        
        image = pipe(
            enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,  # Slightly fewer steps to avoid drift
            guidance_scale=7.0,      # Slightly lower guidance
            height=768,
            width=768,
            generator=generator,
        ).images[0]
        
        story_images.append(image)
        print(f"  ✅ Generated story {i+1}")
    
    return reference_images, story_images

def save_comparison_grid(ref_images, story_images, filename="fixed_comparison.png"):
    """Create comparison grid"""
    all_images = ref_images + story_images
    
    if not all_images:
        return
    
    # Create 2-row grid: references on top, story on bottom
    ref_count = len(ref_images)
    story_count = len(story_images)
    cols = max(ref_count, story_count)
    
    img_width, img_height = all_images[0].size
    grid_width = cols * img_width
    grid_height = 2 * img_height
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Top row: references
    for i, img in enumerate(ref_images):
        grid.paste(img, (i * img_width, 0))
    
    # Bottom row: story
    for i, img in enumerate(story_images):
        grid.paste(img, (i * img_width, img_height))
    
    grid.save(filename)
    print(f"🎨 Comparison grid saved: {filename}")

def main():
    """
    Test the FIXED StoryDiffusion approach
    """
    print("🚀 FIXED STORYDIFFUSION - BETTER PROMPT STRATEGY")
    print("📄 Key insight: Keep prompts CONSISTENT for better results")
    print("🎯 Simple portraits work better than complex scenes")
    print("=" * 65)
    
    # Find model
    possible_paths = [
        "../models/realcartoonxl_v7.safetensors",
        "../../models/realcartoonxl_v7.safetensors",
        "../realcartoonxl_v7.safetensors"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Model not found!")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load pipeline
        print("📦 Loading SDXL pipeline...")
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16
        ).to(device)
        
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()
        
        # Hot-plug StoryDiffusion
        pipe = setup_hotplug_storydiffusion(pipe)
        
        # Test with FIXED strategy
        character_name = "Emma"
        character_desc = "a young woman with long brown hair and blue eyes"
        
        reference_images, story_images = generate_with_fixed_strategy(
            pipe, character_name, character_desc
        )
        
        # Save results
        output_dir = "fixed_strategy_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual images
        for i, img in enumerate(reference_images):
            img.save(f"{output_dir}/fixed_ref_{i+1:02d}.png")
        
        for i, img in enumerate(story_images):
            img.save(f"{output_dir}/fixed_story_{i+1:02d}.png")
        
        # Save comparison grid
        save_comparison_grid(
            reference_images, story_images, 
            f"{output_dir}/fixed_comparison_grid.png"
        )
        
        print(f"\n🎉 FIXED STORYDIFFUSION COMPLETE!")
        print(f"✅ Generated {len(reference_images)} references + {len(story_images)} story images")
        print(f"📁 Saved to: {output_dir}/")
        print(f"\n🎯 KEY IMPROVEMENTS:")
        print(f"  • Consistent portrait framing")
        print(f"  • Same character description with 'exact same' keywords")
        print(f"  • Closer seed values for consistency")
        print(f"  • Stronger consistency strength (0.8)")
        print(f"  • Better negative prompts")
        print(f"\n💡 Story images should now be MUCH more consistent!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()