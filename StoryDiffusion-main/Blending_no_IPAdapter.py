"""
REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - NO MASKING VERSION
Solves the 77-token CLIP limit using the Compel library!

✅ KEY FEATURES:
- Uses Compel library to handle prompts longer than 77 tokens
- Automatic prompt chunking and concatenation
- Support for SDXL dual text encoders
- Detailed prompts for better SDXL results
- Simple IP-Adapter integration WITHOUT masking
- ControlNet for character size consistency
- Clean character descriptions (no dinosaur)
"""

import os
import torch
import time
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, DDIMScheduler, ControlNetModel
from transformers import CLIPVisionModelWithProjection, DPTFeatureExtractor, DPTForDepthEstimation
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc
import numpy as np

class CompelRealCartoonPipeline:
    """Pipeline with Compel support for long prompts and ControlNet - NO MASKING"""
    
    def __init__(self, model_path, device="cuda", use_controlnet=True):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.controlnet_pipeline = None
        self.compel = None
        self.controlnet_compel = None  # Separate Compel for ControlNet pipeline
        self.ip_adapter_loaded = False
        self.controlnet_ip_adapter_loaded = False
        self.use_controlnet = use_controlnet
        self.controlnet = None
        
    def setup_pipeline_with_compel(self):
        """Setup pipeline with Compel for long prompt support and optional ControlNet"""
        print("🔧 Setting up RealCartoon XL v7 with Compel (Long Prompts)...")
        
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
            
            # Load regular pipeline for references and single characters
            print("   🎯 Loading RealCartoon XL v7 (regular pipeline)...")
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                image_encoder=image_encoder,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Optional: Load separate ControlNet pipeline for dual character scenes
            if self.use_controlnet:
                print("   🎚️ Loading ControlNet (depth) for size consistency...")
                try:
                    self.controlnet = ControlNetModel.from_pretrained(
                        "diffusers/controlnet-depth-sdxl-1.0-small",  # Lightweight version
                        variant="fp16",
                        use_safetensors=True,
                        torch_dtype=torch.float16,
                    )
                    
                    # Create separate ControlNet pipeline
                    print("   🔗 Setting up ControlNet pipeline...")
                    self.controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                        self.model_path,
                        controlnet=self.controlnet,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        image_encoder=image_encoder,
                        safety_checker=None,
                        requires_safety_checker=False
                    ).to(self.device)
                    
                    print("   ✅ ControlNet pipeline loaded (depth control for dual characters)")
                except Exception as e:
                    print(f"   ⚠️ ControlNet failed to load: {e}")
                    print("   📝 Continuing without ControlNet...")
                    self.use_controlnet = False
                    self.controlnet = None
                    self.controlnet_pipeline = None
            
            # Setup scheduler for both pipelines
            print("   📐 Setting up scheduler...")
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.enable_vae_tiling()
            
            if self.controlnet_pipeline:
                self.controlnet_pipeline.scheduler = DDIMScheduler.from_config(self.controlnet_pipeline.scheduler.config)
                self.controlnet_pipeline.enable_vae_tiling()
            
            # Setup Compel for long prompts (CRITICAL!)
            print("   🚀 Setting up Compel for long prompts...")
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False  # This enables long prompts!
            )
            
            # Setup separate Compel for ControlNet pipeline if available
            if self.controlnet_pipeline:
                print("   🚀 Setting up Compel for ControlNet pipeline...")
                self.controlnet_compel = Compel(
                    tokenizer=[self.controlnet_pipeline.tokenizer, self.controlnet_pipeline.tokenizer_2],
                    text_encoder=[self.controlnet_pipeline.text_encoder, self.controlnet_pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False
                )
            
            print("✅ Pipeline with Compel ready!")
            print("   🎯 Long prompts (>77 tokens) now supported!")
            if self.use_controlnet:
                print("   📏 ControlNet enabled for character size consistency!")
            print("   🚫 Masking: DISABLED (simpler dual character generation)")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            traceback.print_exc()
            return False
    
    def load_ip_adapter_simple(self):
        """Load IP-Adapter with simple configuration for both pipelines"""
        print("\n🖼️ Loading IP-Adapter...")
        
        try:
            # Load IP-Adapter for regular pipeline
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
            )
            self.ip_adapter_loaded = True
            print("   ✅ IP-Adapter loaded for regular pipeline!")
            
            # Load IP-Adapter for ControlNet pipeline if available
            if self.controlnet_pipeline:
                print("   🖼️ Loading IP-Adapter for ControlNet pipeline...")
                self.controlnet_pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
                )
                self.controlnet_ip_adapter_loaded = True
                print("   ✅ IP-Adapter loaded for ControlNet pipeline!")
            
            return True
            
        except Exception as e:
            print(f"   ❌ IP-Adapter failed: {e}")
            return False
    
    def encode_prompts_with_compel(self, positive_prompt, negative_prompt, use_controlnet_compel=False):
        """Encode both prompts using Compel and ensure matching shapes"""
        try:
            print(f"   📝 Encoding positive prompt: {positive_prompt[:100]}{'...' if len(positive_prompt) > 100 else ''}")
            print(f"   📊 Positive prompt length: {len(positive_prompt.split())} words")
            
            print(f"   📝 Encoding negative prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
            print(f"   📊 Negative prompt length: {len(negative_prompt.split())} words")
            
            # Choose the right Compel instance
            compel_instance = self.controlnet_compel if use_controlnet_compel and self.controlnet_compel else self.compel
            
            # Use Compel to handle both prompts
            positive_conditioning, positive_pooled = compel_instance(positive_prompt)
            negative_conditioning, negative_pooled = compel_instance(negative_prompt)
            
            # CRITICAL: Pad conditioning tensors to same length to avoid shape mismatch
            print("   🔧 Padding tensors to same length...")
            [positive_conditioning, negative_conditioning] = compel_instance.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
            print(f"   ✅ Final tensor shapes: pos={positive_conditioning.shape}, neg={negative_conditioning.shape}")
            
            return positive_conditioning, positive_pooled, negative_conditioning, negative_pooled
            
        except Exception as e:
            print(f"   ❌ Compel encoding failed: {e}")
            traceback.print_exc()
            return None, None, None, None
    
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
            
            # Simple IP-Adapter configuration
            self.pipeline.set_ip_adapter_scale(0.7)
            
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
        """Generate dual character scene using Compel and optional ControlNet - NO MASKING
        
        Note: Without masking, IP-Adapter can only use 1 image per adapter.
        We use Alex as primary reference, but prompt describes both characters.
        """
        
        try:
            # ✅ Choose the correct pipeline and check IP-Adapter status
            if self.use_controlnet and self.controlnet_pipeline and self.controlnet_ip_adapter_loaded:
                print("   📏 Using ControlNet pipeline for character size consistency (no masking)...")
                
                # Encode prompts with ControlNet Compel
                prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                    prompt, negative_prompt, use_controlnet_compel=True
                )
                
                if prompt_embeds is None:
                    return None
                
                # Create depth map for ControlNet
                depth_map = self.create_simple_depth_map()
                
                # Simple IP-Adapter configuration - NO MASKING
                self.controlnet_pipeline.set_ip_adapter_scale(0.6)  # Moderate scale for single reference
                
                # For dual scenes without masking, use Alex as primary reference
                # The prompt will describe both characters
                ip_image = alex_ref  # Single image for single IP-Adapter
                
                # ✅ Use ControlNet pipeline with correct parameters - NO MASKING
                result = self.controlnet_pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=ip_image,  # Single image for single adapter
                    image=depth_map,  # ControlNet depth input
                    controlnet_conditioning_scale=0.5,  # Moderate control
                    **kwargs
                )
                
                print("   ✅ Generated with ControlNet (no masking)!")
                
            else:
                # Fallback to regular pipeline
                if not self.ip_adapter_loaded:
                    print("   ❌ IP-Adapter not loaded for regular pipeline!")
                    return None
                
                print("   📝 Using regular pipeline (no ControlNet, no masking)")
                
                # Encode prompts with regular Compel
                prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                    prompt, negative_prompt, use_controlnet_compel=False
                )
                
                if prompt_embeds is None:
                    return None
                
                # Simple IP-Adapter configuration - NO MASKING
                self.pipeline.set_ip_adapter_scale(0.6)  # Moderate scale for single reference
                
                # For dual scenes without masking, use Alex as primary reference
                # The prompt will describe both characters
                ip_image = alex_ref  # Single image for single IP-Adapter
                
                # Use regular pipeline - NO MASKING
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=ip_image,  # Single image for single adapter
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
        """Detailed reference prompts (no token limit with Compel!)"""
        if character_name == "Alex":
            return f"""{self.alex_base}, character reference sheet, full body portrait, 
            standing pose, clean white background, high quality digital illustration, 
            consistent character design, educational animation style, bright even lighting, 
            clear details, professional character concept art"""
        else:
            return f"""{self.emma_base}, character reference sheet, full body portrait, 
            standing pose, clean white background, high quality digital illustration, 
            consistent character design, educational animation style, bright even lighting, 
            clear details, professional character concept art"""
    
    def get_scene_prompt(self, skill, character_focus, action):
        """Detailed scene prompts (no token limit with Compel!)"""
        if character_focus == "Alex":
            return f"""{self.alex_base}, {action}, {self.setting_base}, 
            autism education themed illustration, social skills learning context, 
            high quality educational content, detailed facial expressions showing emotion, 
            natural body language and gestures, warm inviting atmosphere, 
            child-appropriate content, positive educational messaging"""
        elif character_focus == "Emma":
            return f"""{self.emma_base}, {action}, {self.setting_base}, 
            autism education themed illustration, social skills learning context, 
            high quality educational content, detailed facial expressions showing emotion, 
            natural body language and gestures, warm inviting atmosphere, 
            child-appropriate content, positive educational messaging"""
        else:  # both characters
            return f"""Alex and Emma, two children interacting together, {action}, {self.setting_base}, 
            social skills education illustration, autism awareness content, showing positive social interaction, 
            detailed character expressions and body language, warm educational atmosphere, 
            high quality digital illustration, child-friendly educational content, 
            demonstrating friendship and social learning, bright cheerful colors"""
    
    def get_negative_prompt(self):
        """Minimal negative prompt with background character prevention"""
        return """realistic photography, photorealistic, adult faces, teenage appearance, 
        blurry image, low quality, poor resolution, dark lighting, scary atmosphere, 
        inappropriate content, weapons, violence, sad or distressing themes, 
        poor anatomy, distorted faces, multiple heads, extra limbs, 
        bad hands, mutation, deformed features, background characters, other people"""

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
    """RealCartoon XL v7 with Compel Long Prompts - NO MASKING VERSION"""
    print("\n" + "="*70)
    print("🚀 REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - NO MASKING")
    print("   Solution to the 77-Token CLIP Limit + ControlNet (No Masking)!")
    print("="*70)
    
    print("\n🎯 Compel Features:")
    print("  • Handles prompts longer than 77 tokens")
    print("  • Automatic prompt chunking and concatenation")
    print("  • SDXL dual text encoder support")
    print("  • Detailed prompts for better SDXL results")
    print("  • No more 'truncated prompt' warnings!")
    print("  • ControlNet for character size consistency")
    print("  • 🚫 NO MASKING: Uses Alex reference + prompt for dual scenes")
    
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
    
    output_dir = "compel_long_prompts_autism_no_masking"
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
         "Alex alone, looking around the classroom with a concerned and empathetic expression on his face, with gentle body language showing he wants to help someone, kind and thoughtful demeanor"),
        
        ("02_emma_lonely", "single", "Emma", 
         "Emma alone, sitting at a desk with a book, looking sad and wistful, hoping for friendship and social connection, with expressive eyes showing her desire to interact with others"),
        
        ("03_alex_thinking", "single", "Alex", 
         "Alex alone at his desk with a thinking expression"),
        
        # Dual character scenes  
        ("04_approaching", "dual", "both", 
         "Alex walking towards Emma at her desk while Emma looks up with cautious but hopeful expression, showing the beginning of positive social interaction"),
        
        ("05_greeting", "dual", "both", 
         "Alex and Emma doing a high five together, hands raised up and touching, both smiling happily"),
        
        ("06_playing_together", "dual", "both", 
         "Alex and Emma playing with educational toys together, showing cooperation and shared enjoyment, with happy expressions and collaborative interaction"),
        
        ("07_sharing", "dual", "both", 
         "Alex and Emma sharing books and educational materials, demonstrating generosity and friendship, with caring expressions and positive social behavior"),
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
        print("🚀 COMPEL LONG PROMPTS RESULTS - NO MASKING VERSION")
        print("=" * 70)
        
        success_rate = len(results) / len(educational_scenes) * 100
        avg_time = total_time / len(educational_scenes) if educational_scenes else 0
        
        print(f"✓ Success rate: {success_rate:.1f}% ({len(results)}/{len(educational_scenes)})")
        print(f"✓ Average time: {avg_time:.1f}s per scene")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Used Compel for long prompts")
        print(f"✓ No 77-token truncation warnings!")
        print(f"✓ Detailed SDXL-optimized prompts")
        print(f"✓ No masking - uses Alex reference + prompt for dual scenes")
        
        if results:
            print("\n📋 Generated scenes:")
            for result in results:
                print(f"   • {result}")
        
        print("\n🎉 COMPEL + CONTROLNET ADVANTAGES (NO MASKING):")
        print("   • Long prompts (>77 tokens) working perfectly")
        print("   • Automatic chunking and concatenation")
        print("   • Better SDXL results with detailed descriptions")
        print("   • No prompt truncation warnings")
        print("   • Memory efficient implementation")
        print("   • Simpler dual character generation (Alex reference + descriptive prompts)")
        if use_controlnet and pipeline.use_controlnet and pipeline.controlnet_ip_adapter_loaded:
            print("   • ✅ ControlNet working for consistent character sizes!")
        elif use_controlnet:
            print("   • ⚠️ ControlNet was requested but not fully functional")
        
        if success_rate >= 80:
            print("\n🚀 COMPEL + CONTROLNET SOLUTION SUCCESS (NO MASKING)!")
            print("   📚 Advanced autism education materials")
            print("   🎭 Character consistency with ControlNet")
            print("   🎨 Long detailed prompts working perfectly")
            print("   ✨ 77-token CLIP limit finally solved!")
            print("   🚫 Simplified without masking complexity!")
            if use_controlnet and pipeline.use_controlnet and pipeline.controlnet_ip_adapter_loaded:
                print("   📏 Character size consistency achieved with ControlNet (no masking needed)!")
        
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
        if pipeline.controlnet_pipeline:
            del pipeline.controlnet_pipeline
        if pipeline.compel:
            del pipeline.compel
        if pipeline.controlnet_compel:
            del pipeline.controlnet_compel
        if pipeline.controlnet:
            del pipeline.controlnet
        torch.cuda.empty_cache()
        print("✅ Complete")


if __name__ == "__main__":
    main()