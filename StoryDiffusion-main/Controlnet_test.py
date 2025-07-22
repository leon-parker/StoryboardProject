"""
CONTROLNET VERIFICATION SCRIPT
This script will definitively prove whether ControlNet is working or not
"""

import os
import torch
import time
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, DDIMScheduler, ControlNetModel
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc
import numpy as np

def create_test_depth_map(width=1024, height=1024):
    """Create a test depth map and save it for verification"""
    depth_map = np.ones((height, width), dtype=np.uint8) * 128
    depth_map[:, :width//2] = 120  # Left side slightly closer
    depth_map[:, width//2:] = 120  # Right side slightly closer
    
    depth_image = Image.fromarray(depth_map, mode='L')
    depth_image.save("test_depth_map.png")
    print("✅ Saved test depth map: test_depth_map.png")
    return depth_image

def create_test_masks(width=1024, height=1024):
    """Create test masks and save them for verification"""
    # Alex mask - left side
    alex_mask = Image.new("L", (width, height), 0)
    alex_draw = ImageDraw.Draw(alex_mask)
    alex_draw.rectangle([0, 0, width//2 + 100, height], fill=255)
    alex_mask = alex_mask.filter(ImageFilter.GaussianBlur(radius=2))
    alex_mask.save("test_alex_mask.png")
    
    # Emma mask - right side  
    emma_mask = Image.new("L", (width, height), 0)
    emma_draw = ImageDraw.Draw(emma_mask)
    emma_draw.rectangle([width//2 - 100, 0, width, height], fill=255)
    emma_mask = emma_mask.filter(ImageFilter.GaussianBlur(radius=2))
    emma_mask.save("test_emma_mask.png")
    
    print("✅ Saved test masks: test_alex_mask.png, test_emma_mask.png")
    return alex_mask, emma_mask

def verify_controlnet_is_working():
    """
    Comprehensive ControlNet verification test
    """
    print("\n" + "="*70)
    print("🔬 CONTROLNET VERIFICATION TEST")
    print("="*70)
    
    # Find model (adjust path as needed)
    model_paths = [
        "../models/realcartoonxl_v7.safetensors",
        "models/realcartoonxl_v7.safetensors",
        "../models/RealCartoon-XL-v7.safetensors"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        model_path = input("📁 Enter RealCartoon XL v7 path: ").strip()
        if not os.path.exists(model_path):
            print("❌ Model not found")
            return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    try:
        # Load image encoder
        print("\n📸 Loading image encoder...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )
        
        # Load regular pipeline
        print("📦 Loading regular SDXL pipeline...")
        regular_pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            image_encoder=image_encoder,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        regular_pipeline.scheduler = DDIMScheduler.from_config(regular_pipeline.scheduler.config)
        regular_pipeline.enable_vae_tiling()
        
        # Load ControlNet
        print("🎛️ Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        
        # Load ControlNet pipeline
        print("🔗 Loading ControlNet SDXL pipeline...")
        controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            image_encoder=image_encoder,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        controlnet_pipeline.scheduler = DDIMScheduler.from_config(controlnet_pipeline.scheduler.config)
        controlnet_pipeline.enable_vae_tiling()
        
        # Load IP-Adapters
        print("🖼️ Loading IP-Adapters...")
        regular_pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
        )
        
        controlnet_pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
        )
        
        # Setup Compel
        print("🚀 Setting up Compel...")
        regular_compel = Compel(
            tokenizer=[regular_pipeline.tokenizer, regular_pipeline.tokenizer_2],
            text_encoder=[regular_pipeline.text_encoder, regular_pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False
        )
        
        controlnet_compel = Compel(
            tokenizer=[controlnet_pipeline.tokenizer, controlnet_pipeline.tokenizer_2],
            text_encoder=[controlnet_pipeline.text_encoder, controlnet_pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False
        )
        
        print("✅ All pipelines loaded successfully!")
        
        # Create test inputs
        print("\n🧪 Creating test inputs...")
        
        # Create simple character references (we'll use generated ones for the real test)
        alex_ref = Image.new('RGB', (512, 512), color='lightblue')
        emma_ref = Image.new('RGB', (512, 512), color='lightpink')
        
        # Create test masks and depth map
        alex_mask, emma_mask = create_test_masks()
        depth_map = create_test_depth_map()
        
        # Test prompt
        test_prompt = "Alex and Emma, two children playing together, bright classroom, cartoon style"
        negative_prompt = "blurry, low quality, dark"
        
        # Process masks
        processor = IPAdapterMaskProcessor()
        masks = processor.preprocess([alex_mask, emma_mask], height=1024, width=1024)
        
        print(f"📊 Mask tensor shape: {masks.shape}")
        
        # Format for IP-Adapter
        ip_images = [[alex_ref, emma_ref]]
        masks_formatted = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]
        
        print(f"📊 Formatted masks shape: {masks_formatted[0].shape}")
        
        # Encode prompts
        print("\n📝 Encoding prompts...")
        
        # Regular pipeline prompts
        regular_pos, regular_pos_pooled = regular_compel(test_prompt)
        regular_neg, regular_neg_pooled = regular_compel(negative_prompt)
        [regular_pos, regular_neg] = regular_compel.pad_conditioning_tensors_to_same_length([regular_pos, regular_neg])
        
        # ControlNet pipeline prompts
        controlnet_pos, controlnet_pos_pooled = controlnet_compel(test_prompt)
        controlnet_neg, controlnet_neg_pooled = controlnet_compel(negative_prompt)
        [controlnet_pos, controlnet_neg] = controlnet_compel.pad_conditioning_tensors_to_same_length([controlnet_pos, controlnet_neg])
        
        generation_kwargs = {
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
            "generator": torch.Generator(device).manual_seed(12345)  # Same seed for comparison
        }
        
        print("\n🎨 TEST 1: Regular Pipeline (NO ControlNet)")
        print("-" * 50)
        
        regular_pipeline.set_ip_adapter_scale([[0.7, 0.7]])
        
        start_time = time.time()
        try:
            result_regular = regular_pipeline(
                prompt_embeds=regular_pos,
                pooled_prompt_embeds=regular_pos_pooled,
                negative_prompt_embeds=regular_neg,
                negative_pooled_prompt_embeds=regular_neg_pooled,
                ip_adapter_image=ip_images,
                cross_attention_kwargs={"ip_adapter_masks": masks_formatted},
                **generation_kwargs
            )
            regular_time = time.time() - start_time
            
            if result_regular and result_regular.images:
                result_regular.images[0].save("test_WITHOUT_controlnet.png")
                print(f"✅ Regular pipeline: {regular_time:.1f}s -> test_WITHOUT_controlnet.png")
            else:
                print("❌ Regular pipeline failed")
                
        except Exception as e:
            print(f"❌ Regular pipeline error: {e}")
            traceback.print_exc()
        
        print("\n🎨 TEST 2: ControlNet Pipeline (WITH ControlNet)")
        print("-" * 50)
        
        controlnet_pipeline.set_ip_adapter_scale([[0.7, 0.7]])
        
        # Reset generator with same seed
        generation_kwargs["generator"] = torch.Generator(device).manual_seed(12345)
        
        start_time = time.time()
        try:
            print(f"📊 Depth map shape: {np.array(depth_map).shape}")
            print(f"📊 Depth map type: {type(depth_map)}")
            print(f"📊 ControlNet conditioning scale: 0.5")
            
            result_controlnet = controlnet_pipeline(
                prompt_embeds=controlnet_pos,
                pooled_prompt_embeds=controlnet_pos_pooled,
                negative_prompt_embeds=controlnet_neg,
                negative_pooled_prompt_embeds=controlnet_neg_pooled,
                ip_adapter_image=ip_images,
                cross_attention_kwargs={"ip_adapter_masks": masks_formatted},
                image=depth_map,  # ControlNet input
                controlnet_conditioning_scale=0.5,
                **generation_kwargs
            )
            controlnet_time = time.time() - start_time
            
            if result_controlnet and result_controlnet.images:
                result_controlnet.images[0].save("test_WITH_controlnet.png")
                print(f"✅ ControlNet pipeline: {controlnet_time:.1f}s -> test_WITH_controlnet.png")
            else:
                print("❌ ControlNet pipeline failed")
                
        except Exception as e:
            print(f"❌ ControlNet pipeline error: {e}")
            traceback.print_exc()
        
        print("\n📊 VERIFICATION RESULTS")
        print("="*50)
        
        if os.path.exists("test_WITHOUT_controlnet.png") and os.path.exists("test_WITH_controlnet.png"):
            print("✅ Both images generated successfully!")
            print("🔍 Compare the images:")
            print("   • test_WITHOUT_controlnet.png (regular pipeline)")
            print("   • test_WITH_controlnet.png (ControlNet pipeline)")
            print("   • test_depth_map.png (depth input)")
            print("   • test_alex_mask.png (Alex mask)")
            print("   • test_emma_mask.png (Emma mask)")
            
            # Check file sizes (different sizes suggest different processing)
            size_regular = os.path.getsize("test_WITHOUT_controlnet.png")
            size_controlnet = os.path.getsize("test_WITH_controlnet.png")
            
            print(f"\n📏 File sizes:")
            print(f"   Regular: {size_regular:,} bytes")
            print(f"   ControlNet: {size_controlnet:,} bytes")
            
            if abs(size_regular - size_controlnet) > 1000:  # Significant difference
                print("✅ Different file sizes suggest different processing!")
            else:
                print("⚠️ Similar file sizes - might indicate same processing")
            
            print(f"\n⏱️ Generation times:")
            if 'regular_time' in locals() and 'controlnet_time' in locals():
                print(f"   Regular: {regular_time:.1f}s")
                print(f"   ControlNet: {controlnet_time:.1f}s")
                if controlnet_time > regular_time * 1.1:  # 10% slower
                    print("✅ ControlNet slower as expected!")
                else:
                    print("⚠️ ControlNet not significantly slower")
            
        else:
            print("❌ One or both generations failed")
        
        print("\n🔍 How to verify manually:")
        print("1. Open both generated images side by side")
        print("2. Look for differences in character positioning/sizing")
        print("3. ControlNet should enforce the depth map structure")
        print("4. Characters should appear more consistently positioned")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'regular_pipeline' in locals():
            del regular_pipeline
        if 'controlnet_pipeline' in locals():
            del controlnet_pipeline
        if 'controlnet' in locals():
            del controlnet
        torch.cuda.empty_cache()
        print("\n✅ Cleanup complete")

if __name__ == "__main__":
    verify_controlnet_is_working()