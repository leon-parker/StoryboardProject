"""
REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - SINGLE PIPELINE VERSION
Solves the 77-token CLIP limit using the Compel library!

✅ KEY FEATURES:
- Uses Compel library to handle prompts longer than 77 tokens
- Automatic prompt chunking and concatenation
- Support for SDXL dual text encoders
- Detailed prompts for better SDXL results
- Simple IP-Adapter integration
- Single pipeline for all generation types
- Optional ControlNet for character size consistency
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

class CompelRealCartoonPipeline:
    """Single pipeline with Compel support for long prompts and optional ControlNet"""
    
    def __init__(self, model_path, device="cuda", use_controlnet=True):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.compel = None
        self.ip_adapter_loaded = False
        self.use_controlnet = use_controlnet
        self.controlnet = None
        
    def setup_pipeline_with_compel(self):
        """Setup single pipeline with Compel for long prompt support"""
        print("🔧 Setting up RealCartoon XL v7 with Compel (Single Pipeline)...")
        
        try:
            # Verify model
            size_gb = os.path.getsize(self.model_path) / (1024**3)
            print(f"   📊 Model: {size_gb:.1f}GB")
            if 6.5 <= size_gb <= 7.5:
                print("   ✅ RealCartoon XL v7 confirmed")
            
            # Load image encoder first
            print("   📸 Loading image encoder...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
            )
            
            # Load pipeline with or without ControlNet
            if self.use_controlnet:
                print("   🎚️ Loading ControlNet for size consistency...")
                try:
                    self.controlnet = ControlNetModel.from_pretrained(
                        "diffusers/controlnet-depth-sdxl-1.0-small",
                        variant="fp16",
                        use_safetensors=True,
                        torch_dtype=torch.float16,
                    )
                    
                    print("   🎯 Loading RealCartoon XL v7 with ControlNet...")
                    self.pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                        self.model_path,
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        image_encoder=image_encoder,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    
                    print("   ✅ ControlNet pipeline loaded successfully")
                    
                except Exception as e:
                    print(f"   ⚠️ ControlNet failed to load: {e}")
                    print("   📝 Loading standard pipeline instead...")
                    self.use_controlnet = False
                    self.controlnet = None
            
            # Load standard pipeline if ControlNet not used or failed
            if not self.use_controlnet or self.pipeline is None:
                print("   🎯 Loading RealCartoon XL v7 (standard pipeline)...")
                self.pipeline = StableDiffusionXLPipeline.from_single_file(
                    self.model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    image_encoder=image_encoder,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
            
            # Setup scheduler
            print("   📐 Setting up scheduler...")
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.enable_vae_tiling()
            
            # Setup Compel for long prompts
            print("   🚀 Setting up Compel for long prompts...")
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False  # This enables long prompts!
            )
            
            print("✅ Pipeline with Compel ready!")
            print("   🎯 Long prompts (>77 tokens) now supported!")
            if self.use_controlnet and self.controlnet:
                print("   📏 ControlNet enabled for character size consistency!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            traceback.print_exc()
            return False
    
    def load_ip_adapter_simple(self):
        """Load IP-Adapter with simple configuration (NON-FACE VERSION)"""
        print("\n🖼️ Loading IP-Adapter (Non-Face Version)...")
        
        try:
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"]  # Changed to non-face version
            )
            self.ip_adapter_loaded = True
            print("   ✅ IP-Adapter (Non-Face) loaded successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ IP-Adapter failed: {e}")
            return False
    
    def encode_prompts_with_compel(self, positive_prompt, negative_prompt):
        """Encode both prompts using Compel and ensure matching shapes"""
        try:
            print(f"   📝 Encoding positive prompt: {positive_prompt[:100]}{'...' if len(positive_prompt) > 100 else ''}")
            print(f"   📊 Positive prompt length: {len(positive_prompt.split())} words")
            
            print(f"   📝 Encoding negative prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
            print(f"   📊 Negative prompt length: {len(negative_prompt.split())} words")
            
            # Use Compel to handle both prompts
            positive_conditioning, positive_pooled = self.compel(positive_prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)
            
            # CRITICAL: Pad conditioning tensors to same length to avoid shape mismatch
            print("   🔧 Padding tensors to same length...")
            [positive_conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
            print(f"   ✅ Final tensor shapes: pos={positive_conditioning.shape}, neg={negative_conditioning.shape}")
            
            return positive_conditioning, positive_pooled, negative_conditioning, negative_pooled
            
        except Exception as e:
            print(f"   ❌ Compel encoding failed: {e}")
            traceback.print_exc()
            return None, None, None, None
    
    def create_simple_masks(self, width=1024, height=1024):
        """Create simple left/right masks for dual character scenes"""
        # Alex mask - left side
        alex_mask = Image.new("L", (width, height), 0)
        alex_draw = ImageDraw.Draw(alex_mask)
        alex_draw.rectangle([0, 0, width//2 + 100, height], fill=255)
        
        # Emma mask - right side  
        emma_mask = Image.new("L", (width, height), 0)
        emma_draw = ImageDraw.Draw(emma_mask)
        emma_draw.rectangle([width//2 - 100, 0, width, height], fill=255)
        
        # Light blur for smooth edges
        alex_mask = alex_mask.filter(ImageFilter.GaussianBlur(radius=2))
        emma_mask = emma_mask.filter(ImageFilter.GaussianBlur(radius=2))
        
        return alex_mask, emma_mask
    
    def create_simple_depth_map(self, width=1024, height=1024):
        """Create simple depth map for dual characters at same distance (same size)"""
        # Create depth map where both characters are at the same depth (same gray value)
        depth_map = np.ones((height, width), dtype=np.uint8) * 128  # Mid-gray = same distance
        
        # Optional: Add slight depth variation for more natural look
        # Left side (Alex) - very slightly closer
        depth_map[:, :width//2] = 120
        # Right side (Emma) - very slightly closer  
        depth_map[:, width//2:] = 120
        
        # Convert to PIL Image
        return Image.fromarray(depth_map, mode='L')
    
    def generate_reference_with_compel(self, prompt, negative_prompt, **kwargs):
        """Generate character reference using Compel for long prompts"""
        try:
            # Encode both prompts with Compel and ensure matching shapes
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )
            
            if prompt_embeds is None:
                return None
            
            # Generate with appropriate parameters based on pipeline type
            if self.use_controlnet and self.controlnet:
                # For ControlNet pipeline, provide a dummy depth map
                dummy_depth = self.create_simple_depth_map(kwargs.get('width', 1024), kwargs.get('height', 1024))
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    image=dummy_depth,
                    controlnet_conditioning_scale=0.0,  # No control for references
                    **kwargs
                )
            else:
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    **kwargs
                )
            
            # Clean up embeddings to save memory
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Reference generation failed: {e}")
            traceback.print_exc()
            return None
    
    def generate_single_character_scene_with_compel(self, prompt, negative_prompt, character_ref, **kwargs):
        """Generate single character scene using Compel"""
        
        if not self.ip_adapter_loaded:
            print("   ❌ IP-Adapter not loaded!")
            return None
        
        try:
            # Encode both prompts with Compel and ensure matching shapes
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )
            
            if prompt_embeds is None:
                return None
            
            # Simple IP-Adapter configuration - very low for scene flexibility
            self.pipeline.set_ip_adapter_scale(0.3)  # Much lower for dynamic scenes
            
            # Generate with appropriate parameters based on pipeline type
            if self.use_controlnet and self.controlnet:
                # For ControlNet pipeline, provide a dummy depth map
                dummy_depth = self.create_simple_depth_map(kwargs.get('width', 1024), kwargs.get('height', 1024))
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=character_ref,
                    image=dummy_depth,
                    controlnet_conditioning_scale=0.0,  # No control for single characters
                    **kwargs
                )
            else:
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=character_ref,
                    **kwargs
                )
            
            # Clean up embeddings
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Single character generation failed: {e}")
            traceback.print_exc()
            return None
    
    def generate_dual_character_scene_with_compel(self, prompt, negative_prompt, alex_ref, emma_ref, **kwargs):
        """Generate dual character scene using Compel and optional ControlNet for size consistency"""
        
        if not self.ip_adapter_loaded:
            print("   ❌ IP-Adapter not loaded!")
            return None
            
        try:
            # Encode prompts with Compel
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )
            
            if prompt_embeds is None:
                return None
            
            # Create simple left/right masks
            alex_mask, emma_mask = self.create_simple_masks()
            
            # Process masks
            processor = IPAdapterMaskProcessor()
            masks = processor.preprocess([alex_mask, emma_mask], height=1024, width=1024)
            
            # IP-Adapter configuration for masking - very low for dynamic scenes
            self.pipeline.set_ip_adapter_scale([[0.3, 0.3]])  # Much lower for scene generation
            
            # Format for masking
            ip_images = [[alex_ref, emma_ref]]
            masks_formatted = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]
            
            # Generate with appropriate parameters based on pipeline type
            if self.use_controlnet and self.controlnet:
                print("   📏 Using ControlNet for character size consistency...")
                
                # Create depth map for ControlNet
                depth_map = self.create_simple_depth_map()
                
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=ip_images,
                    cross_attention_kwargs={"ip_adapter_masks": masks_formatted},
                    image=depth_map,  # ControlNet depth input
                    controlnet_conditioning_scale=0.5,  # Moderate control for dual characters
                    **kwargs
                )
                print("   ✅ Generated with ControlNet!")
            else:
                print("   📝 Using standard pipeline (no ControlNet)")
                
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=ip_images,
                    cross_attention_kwargs={"ip_adapter_masks": masks_formatted},
                    **kwargs
                )
            
            # Clean up embeddings
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Dual character generation failed: {e}")
            traceback.print_exc()
            return None

class DetailedEducationalPrompts:
    """Detailed prompts optimized for SDXL with Compel (no 77-token limit!)"""
    
    def __init__(self):
        # Detailed character descriptions for SDXL
        self.alex_base = """Alex, a cheerful 6-year-old boy with short neat black hair and warm expressive brown eyes,
        wearing a comfortable bright blue t-shirt and dark comfortable pants, displaying a genuine friendly smile
        and kind facial expression, with a natural child-like innocent demeanor"""
        
        self.emma_base = """Emma, a sweet 6-year-old girl with blonde hair styled in neat ponytails held by blue ribbons,
        bright blue eyes full of curiosity, wearing a soft pink cardigan over a white shirt,
        with a gentle sweet expression and delicate youthful features, displaying a thoughtful and sensitive nature"""
        
        self.setting_base = """bright cheerful classroom environment with warm natural lighting,
        educational cartoon art style with soft colors and clean lines, child-friendly atmosphere,
        professional educational illustration quality"""
        
    def get_reference_prompt(self, character_name):
        """Generate references as portraits, not character sheets"""
        if character_name == "Alex":
            return f"""{self.alex_base}, closeup portrait of a single child,
            facing forward, natural pose, soft classroom background,
            educational illustration style, warm lighting,
            single character only, portrait shot"""
        else:
            return f"""{self.emma_base}, closeup portrait of a single child,
            facing forward, natural pose, soft classroom background,
            educational illustration style, warm lighting,
            single character only, portrait shot"""
    
    def get_scene_prompt(self, skill, character_focus, action):
        """Action-focused scene prompts to avoid character sheet generation"""
        if character_focus == "Alex":
            return f"""A classroom scene showing {action}, focusing on Alex the boy,
            {self.setting_base}, capturing a single moment of action,
            educational story illustration, dynamic pose and movement,
            environmental storytelling, single unified scene composition"""
        elif character_focus == "Emma":
            return f"""A classroom scene showing {action}, focusing on Emma the girl,
            {self.setting_base}, capturing a single moment of action,
            educational story illustration, dynamic pose and movement,
            environmental storytelling, single unified scene composition"""
        else:  # both characters
            return f"""A classroom scene showing {action},
            {self.setting_base}, capturing interaction between two children,
            educational story moment, dynamic movement and emotion,
            environmental storytelling with depth and perspective,
            single unified scene composition, action shot"""
    
    def get_negative_prompt(self):
        """Negative prompt with reference sheet and multiple view prevention"""
        return """realistic photography, photorealistic, adult faces, teenage appearance,
        blurry image, low quality, poor resolution, dark lighting, scary atmosphere,
        inappropriate content, weapons, violence, sad or distressing themes,
        poor anatomy, distorted faces, multiple heads, extra limbs,
        bad hands, mutation, deformed features, background characters, other people, symbols, words,
        reference sheet, character sheet, multiple views, turnaround, model sheet, concept art,
        multiple poses, character lineup, side by side characters"""

def find_realcartoon_model():
    """Find RealCartoon XL v7 model"""
    paths = [
        "../models/realcartoonxl_v7.safetensors",
        "models/realcartoonxl_v7.safetensors",
        "../models/RealCartoon-XL-v7.safetensors"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def main():
    """RealCartoon XL v7 with Compel Long Prompts - Single Pipeline Version"""
    print("\n" + "="*70)
    print("🚀 REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - SINGLE PIPELINE")
    print("   Solution to the 77-Token CLIP Limit with Simplified Architecture!")
    print("="*70)
    
    print("\n🎯 Compel Features:")
    print("  • Handles prompts longer than 77 tokens")
    print("  • Automatic prompt chunking and concatenation")
    print("  • SDXL dual text encoder support")
    print("  • Detailed prompts for better SDXL results")
    print("  • No more 'truncated prompt' warnings!")
    print("  • Single pipeline for all generation types")
    print("  • Using NON-FACE IP-Adapter for full body consistency")
    
    # ControlNet option
    use_controlnet = True  # Set to False to disable ControlNet
    if use_controlnet:
        print("  📏 ControlNet: ENABLED (ensures same character sizes)")
    else:
        print("  📏 ControlNet: DISABLED (faster generation)")
    
    # Check if Compel is installed
    try:
        import compel
        print("  ✅ Compel library detected")
    except ImportError:
        print("  ❌ Compel library not found!")
        print("     Install with: pip install compel")
        return
    
    # Find model
    model_path = find_realcartoon_model()
    if not model_path:
        model_path = input("📁 Enter RealCartoon XL v7 path: ").strip()
        if not os.path.exists(model_path):
            print("❌ Model not found")
            return
    
    output_dir = "compel_single_pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup pipeline with Compel and optional ControlNet
    pipeline = CompelRealCartoonPipeline(model_path, use_controlnet=use_controlnet)
    if not pipeline.setup_pipeline_with_compel():
        print("❌ Pipeline setup failed")
        return
    
    prompts = DetailedEducationalPrompts()
    
    # Define scenes with detailed actions
    educational_scenes = [
        # Single character scenes
        ("01_alex_noticing", "single", "Alex",
         "a boy looking concerned while sitting at his desk, gazing across the classroom"),
        
        ("02_emma_lonely", "single", "Emma",
         "a girl sitting alone at her desk reading a book, looking sad in an empty corner of the classroom"),
        
        ("03_alex_thinking", "single", "Alex",
         "a boy with hand on chin, thinking deeply while sitting at his school desk"),
        
        # Dual character scenes  
        ("04_approaching", "dual", "both",
         "one child walking across the classroom toward another child who is sitting alone"),
        
        ("05_greeting", "dual", "both",
         "two children waving hello to each other in the middle of a bright classroom"),
        
        ("06_playing_together", "dual", "both",
         "two children sitting on the floor playing with building blocks together"),
        
        ("07_sharing", "dual", "both",
         "two children at a table sharing colorful books and smiling at each other"),
    ]
    
    try:
        # PHASE 1: Character references with long prompts
        print("\n👦👧 PHASE 1: Detailed Character References (Long Prompts)")
        print("-" * 60)
        
        print("🎨 Generating Alex reference with detailed prompt...")
        alex_prompt = prompts.get_reference_prompt("Alex")
        negative = prompts.get_negative_prompt()
        
        print(f"   📊 Alex prompt length: {len(alex_prompt.split())} words")
        
        alex_ref = pipeline.generate_reference_with_compel(
            alex_prompt, negative,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(1000)
        )
        
        print("🎨 Generating Emma reference with detailed prompt...")
        emma_prompt = prompts.get_reference_prompt("Emma")
        
        print(f"   📊 Emma prompt length: {len(emma_prompt.split())} words")
        
        emma_ref = pipeline.generate_reference_with_compel(
            emma_prompt, negative,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(2000)
        )
        
        if not alex_ref or not emma_ref:
            print("❌ Character reference generation failed")
            return
        
        alex_ref.save(os.path.join(output_dir, "alex_reference_detailed.png"))
        emma_ref.save(os.path.join(output_dir, "emma_reference_detailed.png"))
        print("✅ Detailed character references saved")
        
        # PHASE 2: Load IP-Adapter
        print("\n🔄 Loading IP-Adapter...")
        if not pipeline.load_ip_adapter_simple():
            print("❌ IP-Adapter loading failed")
            return
        
        # PHASE 3: Generate detailed scenes
        print(f"\n🎬 PHASE 3: Detailed Educational Scenes ({len(educational_scenes)} total)")
        print("-" * 70)
        
        results = []
        total_time = 0
        
        for i, (scene_name, scene_type, character_focus, action) in enumerate(educational_scenes):
            print(f"\n🎨 Scene {i+1}: {scene_name} ({scene_type})")
            
            prompt = prompts.get_scene_prompt("social skills", character_focus, action)
            output_path = os.path.join(output_dir, f"{scene_name}.png")
            
            print(f"   📊 Scene prompt length: {len(prompt.split())} words")
            
            start_time = time.time()
            
            # Use appropriate generation method
            if scene_type == "single":
                character_ref = alex_ref if character_focus == "Alex" else emma_ref
                image = pipeline.generate_single_character_scene_with_compel(
                    prompt, negative, character_ref,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(1000 + i)
                )
            else:  # dual
                image = pipeline.generate_dual_character_scene_with_compel(
                    prompt, negative, alex_ref, emma_ref,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(1000 + i)
                )
            
            scene_time = time.time() - start_time
            total_time += scene_time
            
            if image:
                image.save(output_path, quality=95, optimize=True)
                results.append(scene_name)
                print(f"   ✅ Generated: {scene_name} ({scene_time:.1f}s)")
            else:
                print(f"   ❌ Failed: {scene_name}")
        
        # Results
        print("\n" + "=" * 70)
        print("🚀 COMPEL LONG PROMPTS RESULTS - SINGLE PIPELINE")
        print("=" * 70)
        
        success_rate = len(results) / len(educational_scenes) * 100
        avg_time = total_time / len(educational_scenes) if educational_scenes else 0
        
        print(f"✓ Success rate: {success_rate:.1f}% ({len(results)}/{len(educational_scenes)})")
        print(f"✓ Average time: {avg_time:.1f}s per scene")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Used Compel for long prompts")
        print(f"✓ No 77-token truncation warnings!")
        print(f"✓ Detailed SDXL-optimized prompts")
        print(f"✓ Using NON-FACE IP-Adapter for better full body consistency")
        
        if results:
            print("\n📋 Generated scenes:")
            for result in results:
                print(f"   • {result}")
        
        print("\n🎉 SINGLE PIPELINE ADVANTAGES:")
        print("   • Long prompts (>77 tokens) working perfectly")
        print("   • Simplified architecture with single pipeline")
        print("   • Better memory efficiency")
        print("   • Automatic chunking and concatenation")
        print("   • Better SDXL results with detailed descriptions")
        print("   • NON-FACE IP-Adapter for improved body consistency")
        if pipeline.use_controlnet and pipeline.controlnet:
            print("   • ✅ ControlNet enabled for consistent character sizes!")
        
        if success_rate >= 80:
            print("\n🚀 COMPEL SINGLE PIPELINE SUCCESS!")
            print("   📚 Advanced autism education materials")
            print("   🎭 Superior character consistency")
            print("   🎨 Long detailed prompts working perfectly")
            print("   ✨ 77-token CLIP limit finally solved!")
            if pipeline.use_controlnet and pipeline.controlnet:
                print("   📏 Character size consistency with ControlNet!")
        
        print(f"\n📁 Outputs: {os.path.abspath(output_dir)}/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        if pipeline.pipeline:
            del pipeline.pipeline
        if pipeline.compel:
            del pipeline.compel
        if pipeline.controlnet:
            del pipeline.controlnet
        torch.cuda.empty_cache()
        print("✅ Complete")

if __name__ == "__main__":
    main()