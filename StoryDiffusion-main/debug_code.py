# debug_storydiffusion.py
"""
Debug StoryDiffusion - Find Why Story Mode Breaks
Let's isolate the problem step by step
"""

import sys
import os
import torch
import numpy as np
import random
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test_baseline_generation(pipe, character_desc):
    """
    Test 1: Generate WITHOUT any StoryDiffusion to establish baseline
    """
    print("\n🧪 TEST 1: BASELINE GENERATION (No StoryDiffusion)")
    print("Testing if the pipeline works normally without any modifications")
    
    baseline_images = []
    
    # Test prompts
    test_prompts = [
        f"anime portrait of {character_desc}, wearing school uniform",
        f"anime portrait of {character_desc}, wearing casual dress", 
        f"anime portrait of {character_desc}, wearing winter coat"
    ]
    
    for i, prompt in enumerate(test_prompts):
        setup_seed(12345 + i * 100)
        generator = torch.Generator(device=pipe.device).manual_seed(12345 + i * 100)
        
        enhanced_prompt = f"masterpiece, best quality, {prompt}, anime art style"
        negative_prompt = "photorealistic, realistic, blurry, low quality, bad anatomy"
        
        print(f"  🎨 Baseline {i+1}: {prompt[:50]}...")
        
        try:
            image = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=1024,
                width=1024,
                generator=generator,
            ).images[0]
            
            baseline_images.append(image)
            print(f"  ✅ Baseline {i+1} - SUCCESS")
            
        except Exception as e:
            print(f"  ❌ Baseline {i+1} - FAILED: {e}")
            return None
    
    print(f"✅ Baseline test passed - pipeline works normally")
    return baseline_images

def test_reference_only(pipe, character_desc):
    """
    Test 2: Generate references and check if they work
    """
    print("\n🧪 TEST 2: REFERENCE GENERATION ONLY")
    print("Testing reference generation with StoryDiffusion enabled")
    
    # Setup minimal StoryDiffusion (just for testing)
    try:
        sys.path.insert(0, '.')
        from utils.gradio_utils import SpatialAttnProcessor2_0
        
        # Apply to minimal set of layers for testing
        attn_procs = {}
        applied_count = 0
        
        for name in pipe.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") 
                else pipe.unet.config.cross_attention_dim
            )
            
            # Only apply to a few up-block layers for testing
            if (cross_attention_dim is None and 
                "up_blocks.3" in name):  # Only the last up-block
                
                attn_procs[name] = SpatialAttnProcessor2_0(
                    hidden_size=pipe.unet.config.block_out_channels[-1],
                    cross_attention_dim=cross_attention_dim
                )
                applied_count += 1
            else:
                attn_procs[name] = pipe.unet.attn_processors[name]
        
        pipe.unet.set_attn_processor(attn_procs)
        print(f"  🔌 Applied StoryDiffusion to {applied_count} layers (minimal test)")
        
        # Set reference mode
        import utils.gradio_utils as gradio_utils
        gradio_utils.write = True  # Reference mode
        gradio_utils.cur_step = 0
        gradio_utils.attn_count = 0
        
    except ImportError:
        print("  ⚠️ Official StoryDiffusion not available - skipping")
        return None
    
    reference_images = []
    
    # Generate references
    ref_prompts = [
        f"anime portrait of {character_desc}, facing forward, neutral expression",
        f"anime portrait of {character_desc}, gentle smile",
        f"anime portrait of {character_desc}, happy expression"
    ]
    
    for i, prompt in enumerate(ref_prompts):
        setup_seed(12345 + i * 100)
        generator = torch.Generator(device=pipe.device).manual_seed(12345 + i * 100)
        
        enhanced_prompt = f"masterpiece, best quality, {prompt}, anime art style"
        negative_prompt = "photorealistic, realistic, blurry, low quality, bad anatomy"
        
        print(f"  🎨 Reference {i+1}: {prompt[:50]}...")
        
        try:
            # Reset gradio utils for each generation
            gradio_utils.cur_step = 0
            gradio_utils.attn_count = 0
            
            image = pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=1024,
                width=1024,
                generator=generator,
            ).images[0]
            
            reference_images.append(image)
            print(f"  ✅ Reference {i+1} - SUCCESS")
            
        except Exception as e:
            print(f"  ❌ Reference {i+1} - FAILED: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print(f"✅ Reference test passed - {len(reference_images)} images generated")
    return reference_images

def test_story_mode_switch(pipe, character_desc):
    """
    Test 3: Switch to story mode and see if it breaks
    """
    print("\n🧪 TEST 3: STORY MODE SWITCH")
    print("Testing what happens when we switch to story mode")
    
    try:
        import utils.gradio_utils as gradio_utils
        
        # Switch to story mode
        print("  🔄 Switching to story mode...")
        gradio_utils.write = False  # Story mode
        gradio_utils.cur_step = 0
        gradio_utils.attn_count = 0
        
        print("  📊 Checking gradio_utils state:")
        print(f"    write: {gradio_utils.write}")
        print(f"    cur_step: {gradio_utils.cur_step}")
        print(f"    attn_count: {gradio_utils.attn_count}")
        
        # Try to generate ONE story image
        setup_seed(22345)
        generator = torch.Generator(device=pipe.device).manual_seed(22345)
        
        prompt = f"anime portrait of the same {character_desc}, wearing school uniform"
        enhanced_prompt = f"masterpiece, best quality, {prompt}, anime art style"
        negative_prompt = "photorealistic, realistic, blurry, low quality, bad anatomy, different person"
        
        print(f"  🎨 Story test: {prompt[:50]}...")
        
        # Reset for story generation
        gradio_utils.cur_step = 0
        gradio_utils.attn_count = 0
        
        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=1024,
            width=1024,
            generator=generator,
        ).images[0]
        
        print(f"  ✅ Story mode - SUCCESS")
        return image
        
    except Exception as e:
        print(f"  ❌ Story mode - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_cleanup(pipe):
    """
    Test 4: Check if memory/cache cleanup helps
    """
    print("\n🧪 TEST 4: MEMORY CLEANUP")
    print("Testing if clearing cache fixes story mode")
    
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  ✅ Cleared CUDA cache")
        
        # Reset UNet to default processors
        print("  🔄 Resetting UNet processors to default...")
        from diffusers.models.attention_processor import AttnProcessor2_0
        
        # Reset all processors to default
        default_procs = {}
        for name in pipe.unet.attn_processors.keys():
            default_procs[name] = AttnProcessor2_0()
        
        pipe.unet.set_attn_processor(default_procs)
        print("  ✅ Reset to default processors")
        
        # Test generation after cleanup
        setup_seed(33345)
        generator = torch.Generator(device=pipe.device).manual_seed(33345)
        
        prompt = "masterpiece, best quality, anime portrait of a beautiful girl, cute expression"
        
        print("  🎨 Testing after cleanup...")
        
        image = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            num_inference_steps=25,
            guidance_scale=7.5,
            height=1024,
            width=1024,
            generator=generator,
        ).images[0]
        
        print("  ✅ Post-cleanup generation - SUCCESS")
        return image
        
    except Exception as e:
        print(f"  ❌ Post-cleanup generation - FAILED: {e}")
        return None

def test_gradio_utils_state():
    """
    Test 5: Check gradio_utils internal state
    """
    print("\n🧪 TEST 5: GRADIO_UTILS STATE INSPECTION")
    
    try:
        import utils.gradio_utils as gradio_utils
        
        print("  📊 Current gradio_utils state:")
        
        # Check all available attributes
        attrs_to_check = ['write', 'cur_step', 'attn_count', 'total_count', 'height', 'width', 
                         'sa32', 'sa64', 'sa16', 'mask1024', 'mask4096', 'id_length', 
                         'character_dict', 'cur_character', 'ref_indexs_dict', 'ref_totals']
        
        for attr in attrs_to_check:
            if hasattr(gradio_utils, attr):
                value = getattr(gradio_utils, attr)
                print(f"    {attr}: {value}")
        
        # Check if any processors have id_bank
        print("  🏦 Checking for id_bank in processors...")
        for name, processor in gradio_utils.__dict__.items():
            if hasattr(processor, 'id_bank') if isinstance(processor, object) else False:
                print(f"    Found id_bank in {name}: {len(processor.id_bank) if processor.id_bank else 'empty'}")
        
    except ImportError:
        print("  ⚠️ gradio_utils not available")
    except Exception as e:
        print(f"  ❌ State inspection failed: {e}")

def main():
    """
    Run all debug tests to find why story mode breaks
    """
    print("🔍 STORYDIFFUSION DEBUG SESSION")
    print("Finding why story images break while references work")
    print("=" * 60)
    
    # Find model
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
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup clean pipeline
    print(f"📦 Loading clean pipeline: {os.path.basename(model_path)}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)
    
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    
    character_desc = "a beautiful anime girl with long brown hair, bright blue eyes, soft facial features"
    
    # Run debug tests
    print(f"🎭 Test character: {character_desc}")
    
    # Test 1: Baseline
    baseline_images = test_baseline_generation(pipe, character_desc)
    
    # Test 2: References
    reference_images = test_reference_only(pipe, character_desc)
    
    # Test 3: Story mode switch
    story_image = test_story_mode_switch(pipe, character_desc)
    
    # Test 4: Memory cleanup
    cleanup_image = test_memory_cleanup(pipe)
    
    # Test 5: State inspection
    test_gradio_utils_state()
    
    # Save debug results
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    if baseline_images:
        for i, img in enumerate(baseline_images):
            img.save(f"{output_dir}/debug_baseline_{i+1}.png")
        print(f"💾 Saved {len(baseline_images)} baseline images")
    
    if reference_images:
        for i, img in enumerate(reference_images):
            img.save(f"{output_dir}/debug_reference_{i+1}.png")
        print(f"💾 Saved {len(reference_images)} reference images")
    
    if story_image:
        story_image.save(f"{output_dir}/debug_story.png")
        print(f"💾 Saved story image")
    
    if cleanup_image:
        cleanup_image.save(f"{output_dir}/debug_cleanup.png")
        print(f"💾 Saved cleanup image")
    
    # Summary
    print(f"\n📋 DEBUG SUMMARY:")
    print(f"  Baseline generation: {'✅ PASS' if baseline_images else '❌ FAIL'}")
    print(f"  Reference generation: {'✅ PASS' if reference_images else '❌ FAIL'}")
    print(f"  Story mode switch: {'✅ PASS' if story_image else '❌ FAIL'}")
    print(f"  Memory cleanup: {'✅ PASS' if cleanup_image else '❌ FAIL'}")
    
    if baseline_images and not story_image:
        print(f"\n🎯 DIAGNOSIS: StoryDiffusion breaks story mode generation")
        print(f"   - Baseline works → Pipeline is fine")
        print(f"   - References work → StoryDiffusion setup is fine")
        print(f"   - Story fails → Problem in story mode logic")
    elif not baseline_images:
        print(f"\n🎯 DIAGNOSIS: Pipeline/model issue (not StoryDiffusion)")
    else:
        print(f"\n🎯 DIAGNOSIS: Everything works - investigate prompt/seed differences")

if __name__ == "__main__":
    main()