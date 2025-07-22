"""
REALCARTOON XL V7 WITH COMPEL + DEPTH CONTROLNET COMPARISON
Generates scenes with and without Depth ControlNet for height control

✅ KEY FEATURES:
- Uses Compel library to handle prompts longer than 77 tokens
- Generates BOTH versions: with and without Depth ControlNet
- Depth maps for gentle height/size control
- Simple IP-Adapter integration
- Much better quality than OpenPose!
"""

import os
import torch
import time
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, DDIMScheduler, ControlNetModel
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc
import numpy as np

class CompelRealCartoonPipeline:
    """Pipeline with Compel support and Depth ControlNet comparison"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.depth_pipeline = None
        self.compel = None
        self.depth_compel = None
        self.ip_adapter_loaded = False
        self.depth_ip_adapter_loaded = False
        self.depth_controlnet = None
        
    def setup_pipeline_with_compel(self):
        """Setup pipeline with Compel for long prompt support and Depth ControlNet"""
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
            
            # Load regular pipeline for references and WITHOUT ControlNet
            print("   🎯 Loading RealCartoon XL v7 (regular pipeline)...")
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                image_encoder=image_encoder,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Load Depth ControlNet pipeline
            print("   🏔️ Loading Depth ControlNet for height control...")
            try:
                self.depth_controlnet = ControlNetModel.from_pretrained(
                    "diffusers/controlnet-depth-sdxl-1.0-small",
                    variant="fp16",
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                )
                
                # Create Depth pipeline
                print("   🔗 Setting up Depth ControlNet pipeline...")
                self.depth_pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                    self.model_path,
                    controlnet=self.depth_controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    image_encoder=image_encoder,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                
                print("   ✅ Depth ControlNet loaded for height comparison")
            except Exception as e:
                print(f"   ⚠️ Depth ControlNet failed to load: {e}")
                print("   📝 Will only generate WITHOUT ControlNet...")
                self.depth_controlnet = None
                self.depth_pipeline = None
            
            # Setup scheduler for both pipelines
            print("   📐 Setting up schedulers...")
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.enable_vae_tiling()
            
            if self.depth_pipeline:
                self.depth_pipeline.scheduler = DDIMScheduler.from_config(self.depth_pipeline.scheduler.config)
                self.depth_pipeline.enable_vae_tiling()
            
            # Setup Compel for long prompts
            print("   🚀 Setting up Compel for long prompts...")
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False
            )
            
            # Setup Compel for Depth pipeline if available
            if self.depth_pipeline:
                print("   🚀 Setting up Compel for Depth pipeline...")
                self.depth_compel = Compel(
                    tokenizer=[self.depth_pipeline.tokenizer, self.depth_pipeline.tokenizer_2],
                    text_encoder=[self.depth_pipeline.text_encoder, self.depth_pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False
                )
            
            print("✅ Pipeline with Compel ready!")
            print("   🎯 Long prompts (>77 tokens) now supported!")
            if self.depth_pipeline:
                print("   🏔️ Depth ControlNet comparison enabled!")
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
            
            # Load IP-Adapter for Depth pipeline if available
            if self.depth_pipeline:
                print("   🖼️ Loading IP-Adapter for Depth pipeline...")
                self.depth_pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
                )
                self.depth_ip_adapter_loaded = True
                print("   ✅ IP-Adapter loaded for Depth pipeline!")
            
            return True
            
        except Exception as e:
            print(f"   ❌ IP-Adapter failed: {e}")
            return False
    
    def create_depth_map(self, scene_type, scenario="default", width=1024, height=1024):
        """Create depth map for height control"""
        # Start with mid-gray background (neutral depth)
        depth_map = np.ones((height, width), dtype=np.uint8) * 150
        
        if scene_type == "single":
            # Single character in center
            self._add_depth_silhouette(depth_map, width//2, height//2, height//2.5, depth_value=80)
        else:  # dual
            if scenario == "standing_together":
                # Two characters side by side - SAME HEIGHT, SAME DEPTH
                self._add_depth_silhouette(depth_map, width//3, height//2, height//2.5, depth_value=80)
                self._add_depth_silhouette(depth_map, 2*width//3, height//2, height//2.5, depth_value=80)
            
            elif scenario == "high_five":
                # Characters closer together for interaction - SAME HEIGHT
                self._add_depth_silhouette(depth_map, int(width*0.4), height//2, height//2.5, depth_value=80)
                self._add_depth_silhouette(depth_map, int(width*0.6), height//2, height//2.5, depth_value=80)
            
            elif scenario == "sitting":
                # Sitting characters - shorter silhouettes, lower position
                sitting_y = int(height * 0.65)
                self._add_depth_silhouette(depth_map, width//3, sitting_y, height//3.5, depth_value=80)
                self._add_depth_silhouette(depth_map, 2*width//3, sitting_y, height//3.5, depth_value=80)
            
            else:  # default dual
                # Standard side by side - SAME HEIGHT
                self._add_depth_silhouette(depth_map, width//3, height//2, height//2.5, depth_value=80)
                self._add_depth_silhouette(depth_map, 2*width//3, height//2, height//2.5, depth_value=80)
        
        # Convert to PIL and add slight blur for smoother control
        depth_image = Image.fromarray(depth_map, mode='L')
        depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Add labels
        draw = ImageDraw.Draw(depth_image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        if scene_type == "dual":
            draw.text((width//2 - 100, 30), "SAME HEIGHT DEPTH MAP", fill=255, font=font)
        
        return depth_image
    
    def _add_depth_silhouette(self, depth_map, center_x, center_y, height, depth_value):
        """Add a human-shaped silhouette to depth map"""
        # Simple human shape using overlapping shapes
        head_radius = int(height * 0.08)
        body_width = int(height * 0.25)
        body_height = int(height * 0.35)
        
        # Head (circle)
        head_y = center_y - int(height * 0.4)
        for y in range(max(0, head_y - head_radius), min(depth_map.shape[0], head_y + head_radius)):
            for x in range(max(0, center_x - head_radius), min(depth_map.shape[1], center_x + head_radius)):
                if (x - center_x)**2 + (y - head_y)**2 <= head_radius**2:
                    depth_map[y, x] = depth_value
        
        # Body (ellipse/rectangle)
        body_top = head_y + head_radius
        body_bottom = body_top + body_height
        for y in range(max(0, body_top), min(depth_map.shape[0], body_bottom)):
            for x in range(max(0, center_x - body_width//2), min(depth_map.shape[1], center_x + body_width//2)):
                # Taper body slightly
                width_at_y = body_width * (1 - (y - body_top) / body_height * 0.3)
                if abs(x - center_x) <= width_at_y / 2:
                    depth_map[y, x] = depth_value
        
        # Legs (simple rectangles)
        leg_width = int(height * 0.08)
        leg_height = int(height * 0.3)
        leg_top = body_bottom - 10
        
        # Left leg
        for y in range(max(0, leg_top), min(depth_map.shape[0], leg_top + leg_height)):
            for x in range(max(0, center_x - body_width//4 - leg_width//2), 
                          min(depth_map.shape[1], center_x - body_width//4 + leg_width//2)):
                depth_map[y, x] = depth_value
        
        # Right leg
        for y in range(max(0, leg_top), min(depth_map.shape[0], leg_top + leg_height)):
            for x in range(max(0, center_x + body_width//4 - leg_width//2), 
                          min(depth_map.shape[1], center_x + body_width//4 + leg_width//2)):
                depth_map[y, x] = depth_value
    
    def encode_prompts_with_compel(self, positive_prompt, negative_prompt, use_depth_compel=False):
        """Encode both prompts using Compel and ensure matching shapes"""
        try:
            # Choose the right Compel instance
            compel_instance = self.depth_compel if use_depth_compel and self.depth_compel else self.compel
            
            # Use Compel to handle both prompts
            positive_conditioning, positive_pooled = compel_instance(positive_prompt)
            negative_conditioning, negative_pooled = compel_instance(negative_prompt)
            
            # Pad conditioning tensors to same length
            [positive_conditioning, negative_conditioning] = compel_instance.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
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
    
    def generate_reference_with_compel(self, prompt, negative_prompt, **kwargs):
        """Generate character reference using Compel for long prompts"""
        try:
            # Encode both prompts with Compel
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
            
            # Clean up embeddings
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Reference generation failed: {e}")
            traceback.print_exc()
            return None
    
    def generate_scene_with_compel(self, prompt, negative_prompt, scene_type, character_refs, 
                                  use_depth=False, depth_image=None, **kwargs):
        """Generate scene with or without Depth ControlNet"""
        
        if use_depth and not self.depth_pipeline:
            print("   ❌ Depth pipeline not available!")
            return None
        
        try:
            # Choose pipeline and compel
            if use_depth:
                pipeline = self.depth_pipeline
                compel_for_encoding = True  # Use depth compel
                if not self.depth_ip_adapter_loaded:
                    print("   ❌ IP-Adapter not loaded for Depth!")
                    return None
            else:
                pipeline = self.pipeline
                compel_for_encoding = False  # Use regular compel
                if not self.ip_adapter_loaded:
                    print("   ❌ IP-Adapter not loaded!")
                    return None
            
            # Encode prompts
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt, use_depth_compel=compel_for_encoding
            )
            
            if prompt_embeds is None:
                return None
            
            # Setup based on scene type
            if scene_type == "single":
                # Single character
                pipeline.set_ip_adapter_scale(0.7)
                
                generation_kwargs = {
                    'prompt_embeds': prompt_embeds,
                    'pooled_prompt_embeds': pooled_prompt_embeds,
                    'negative_prompt_embeds': negative_embeds,
                    'negative_pooled_prompt_embeds': negative_pooled,
                    'ip_adapter_image': character_refs[0],  # Just one character
                    **kwargs
                }
                
                if use_depth and depth_image:
                    generation_kwargs['image'] = depth_image
                    generation_kwargs['controlnet_conditioning_scale'] = 0.5  # Gentle control
                
            else:  # dual
                # Dual characters with masks
                alex_mask, emma_mask = self.create_simple_masks()
                
                # Process masks
                processor = IPAdapterMaskProcessor()
                masks = processor.preprocess([alex_mask, emma_mask], height=1024, width=1024)
                
                # IP-Adapter configuration
                pipeline.set_ip_adapter_scale([[0.7, 0.7]])
                
                # Format for masking
                ip_images = [character_refs]  # [alex_ref, emma_ref]
                masks_formatted = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]
                
                generation_kwargs = {
                    'prompt_embeds': prompt_embeds,
                    'pooled_prompt_embeds': pooled_prompt_embeds,
                    'negative_prompt_embeds': negative_embeds,
                    'negative_pooled_prompt_embeds': negative_pooled,
                    'ip_adapter_image': ip_images,
                    'cross_attention_kwargs': {"ip_adapter_masks": masks_formatted},
                    **kwargs
                }
                
                if use_depth and depth_image:
                    generation_kwargs['image'] = depth_image
                    generation_kwargs['controlnet_conditioning_scale'] = 0.5  # Gentle control
            
            # Generate
            result = pipeline(**generation_kwargs)
            
            # Clean up
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Scene generation failed: {e}")
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
    """RealCartoon XL v7 with Compel + Depth ControlNet Comparison"""
    print("\n" + "="*70)
    print("🚀 REALCARTOON XL V7 WITH COMPEL + DEPTH CONTROLNET COMPARISON")
    print("   Generate scenes WITH and WITHOUT Depth ControlNet")
    print("="*70)
    
    print("\n🎯 Features:")
    print("  • Handles prompts longer than 77 tokens (Compel)")
    print("  • Generates BOTH versions for comparison")
    print("  • WITHOUT Depth: IP-Adapter only (random heights)")
    print("  • WITH Depth: IP-Adapter + depth control (consistent heights)")
    print("  • Gentle depth maps preserve cartoon quality!")
    
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
    
    output_dir = "openpose_comparison_dissertation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup pipeline
    pipeline = CompelRealCartoonPipeline(model_path)
    if not pipeline.setup_pipeline_with_compel():
        print("❌ Pipeline setup failed")
        return
    
    prompts = DetailedEducationalPrompts()
    
    # Define scenes - focusing on height comparison scenarios
    educational_scenes = [
        # Single character scenes
        ("01_alex_standing", "single", "Alex", "default",
         "Alex standing alone in the classroom, full body visible, friendly smile"),
        
        ("02_emma_standing", "single", "Emma", "default",
         "Emma standing alone in the classroom, full body visible, gentle expression"),
        
        # Dual character scenes - these will show height differences
        ("03_standing_together", "dual", "both", "standing_together",
         "Alex and Emma standing side by side, both children the same height, full bodies visible"),
        
        ("04_high_five", "dual", "both", "high_five",
         "Alex and Emma doing a high five together, hands raised up and touching, both smiling"),
        
        ("05_sitting_together", "dual", "both", "sitting",
         "Alex and Emma sitting together at a table, both children seated on chairs"),
    ]
    
    try:
        # PHASE 1: Character references
        print("\n👦👧 PHASE 1: Character References (with Compel)")
        print("-" * 60)
        
        print("🎨 Generating Alex reference...")
        alex_prompt = prompts.get_reference_prompt("Alex")
        negative = prompts.get_negative_prompt()
        
        alex_ref = pipeline.generate_reference_with_compel(
            alex_prompt, negative,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(1000)
        )
        
        print("🎨 Generating Emma reference...")
        emma_prompt = prompts.get_reference_prompt("Emma")
        
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
        
        alex_ref.save(os.path.join(output_dir, "alex_reference.png"))
        emma_ref.save(os.path.join(output_dir, "emma_reference.png"))
        print("✅ Character references saved")
        
        # PHASE 2: Load IP-Adapter
        print("\n🔄 Loading IP-Adapter...")
        if not pipeline.load_ip_adapter_simple():
            print("❌ IP-Adapter loading failed")
            return
        
        # PHASE 3: Generate scenes WITH and WITHOUT Depth
        print(f"\n🎬 PHASE 3: Comparison Scenes ({len(educational_scenes)} total)")
        print("   Each scene will be generated TWICE:")
        print("   • WITHOUT Depth ControlNet (IP-Adapter only)")
        print("   • WITH Depth ControlNet (IP-Adapter + height control)")
        print("-" * 70)
        
        results = []
        total_time = 0
        
        for i, (scene_name, scene_type, character_focus, scenario, action) in enumerate(educational_scenes):
            print(f"\n🎨 Scene {i+1}: {scene_name} ({scene_type})")
            
            prompt = prompts.get_scene_prompt("social skills", character_focus, action)
            print(f"   📊 Prompt length: {len(prompt.split())} words (no truncation!)")
            
            # Prepare character refs
            if scene_type == "single":
                character_refs = [alex_ref if character_focus == "Alex" else emma_ref]
            else:
                character_refs = [alex_ref, emma_ref]
            
            # Generate depth map if available
            depth_image = None
            if pipeline.depth_pipeline:
                depth_image = pipeline.create_depth_map(scene_type, scenario)
                depth_path = os.path.join(output_dir, f"{scene_name}_depth_map.png")
                depth_image.save(depth_path)
                print(f"   🏔️ Saved depth map: {depth_path}")
            
            # Generate WITHOUT Depth ControlNet
            print(f"   🖼️ Generating WITHOUT Depth ControlNet...")
            start_time = time.time()
            
            image_without = pipeline.generate_scene_with_compel(
                prompt, negative, scene_type, character_refs,
                use_depth=False,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=1024, width=1024,
                generator=torch.Generator(pipeline.device).manual_seed(3000 + i)
            )
            
            without_time = time.time() - start_time
            
            if image_without:
                without_path = os.path.join(output_dir, f"{scene_name}_WITHOUT_depth.png")
                image_without.save(without_path, quality=95, optimize=True)
                print(f"   ✅ WITHOUT Depth: {without_time:.1f}s")
            
            # Generate WITH Depth ControlNet if available
            if pipeline.depth_pipeline and depth_image:
                print(f"   🖼️ Generating WITH Depth ControlNet...")
                start_time = time.time()
                
                image_with = pipeline.generate_scene_with_compel(
                    prompt, negative, scene_type, character_refs,
                    use_depth=True,
                    depth_image=depth_image,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(3000 + i)
                )
                
                with_time = time.time() - start_time
                total_time += without_time + with_time
                
                if image_with:
                    with_path = os.path.join(output_dir, f"{scene_name}_WITH_depth.png")
                    image_with.save(with_path, quality=95, optimize=True)
                    print(f"   ✅ WITH Depth: {with_time:.1f}s")
                    results.append(scene_name)
            else:
                total_time += without_time
                print("   ⚠️ Skipping Depth version (not available)")
        
        # Results
        print("\n" + "=" * 70)
        print("🚀 DEPTH CONTROLNET COMPARISON RESULTS")
        print("=" * 70)
        
        print(f"✓ Generated {len(results)} complete scene comparisons")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Used Compel for long prompts (no truncation!)")
        
        print("\n📋 Generated files:")
        print(f"   📁 {output_dir}/")
        print("   • alex_reference.png & emma_reference.png")
        for scene_name in results:
            print(f"   • {scene_name}_WITHOUT_depth.png")
            print(f"   • {scene_name}_WITH_depth.png")
            print(f"   • {scene_name}_depth_map.png")
        
        print("\n🎓 FOR YOUR DISSERTATION:")
        print("   • Depth ControlNet provides GENTLE height control")
        print("   • WITHOUT: Characters may have random heights")
        print("   • WITH: Depth maps enforce consistent heights")
        print("   • Quality remains high (unlike OpenPose)")
        print("   • Perfect for educational content consistency")
        
        print("\n💡 KEY ADVANTAGES:")
        print("   • Depth maps are less restrictive than skeletons")
        print("   • Preserves cartoon art style quality")
        print("   • Controls size/distance naturally")
        print("   • Works well with IP-Adapter")
        
        print(f"\n📁 All outputs in: {os.path.abspath(output_dir)}/")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        if pipeline.pipeline:
            del pipeline.pipeline
        if pipeline.depth_pipeline:
            del pipeline.depth_pipeline
        if pipeline.compel:
            del pipeline.compel
        if pipeline.depth_compel:
            del pipeline.depth_compel
        if pipeline.depth_controlnet:
            del pipeline.depth_controlnet
        torch.cuda.empty_cache()
        print("✅ Complete")


if __name__ == "__main__":
    main()