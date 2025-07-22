"""
REALCARTOON XL V7 WITH COMPEL + OPENPOSE COMPARISON
Generates scenes with and without OpenPose ControlNet for dissertation comparison

✅ KEY FEATURES:
- Uses Compel library to handle prompts longer than 77 tokens
- Generates BOTH versions: with and without OpenPose
- Support for SDXL dual text encoders
- Simple IP-Adapter integration
- OpenPose for height/pose control comparison
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
    """Pipeline with Compel support and OpenPose ControlNet comparison"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.openpose_pipeline = None
        self.compel = None
        self.openpose_compel = None
        self.ip_adapter_loaded = False
        self.openpose_ip_adapter_loaded = False
        self.openpose_controlnet = None
        
    def setup_pipeline_with_compel(self):
        """Setup pipeline with Compel for long prompt support and OpenPose ControlNet"""
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
            
            # Load regular pipeline for references and WITHOUT OpenPose
            print("   🎯 Loading RealCartoon XL v7 (regular pipeline)...")
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                image_encoder=image_encoder,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Load OpenPose ControlNet pipeline
            print("   🦴 Loading OpenPose ControlNet for comparison...")
            try:
                self.openpose_controlnet = ControlNetModel.from_pretrained(
                    "thibaud/controlnet-openpose-sdxl-1.0",
                    torch_dtype=torch.float16,
                )
                
                # Create OpenPose pipeline
                print("   🔗 Setting up OpenPose pipeline...")
                self.openpose_pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                    self.model_path,
                    controlnet=self.openpose_controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    image_encoder=image_encoder,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                
                print("   ✅ OpenPose pipeline loaded for comparison")
            except Exception as e:
                print(f"   ⚠️ OpenPose ControlNet failed to load: {e}")
                print("   📝 Will only generate WITHOUT OpenPose...")
                self.openpose_controlnet = None
                self.openpose_pipeline = None
            
            # Setup scheduler for both pipelines
            print("   📐 Setting up schedulers...")
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.enable_vae_tiling()
            
            if self.openpose_pipeline:
                self.openpose_pipeline.scheduler = DDIMScheduler.from_config(self.openpose_pipeline.scheduler.config)
                self.openpose_pipeline.enable_vae_tiling()
            
            # Setup Compel for long prompts
            print("   🚀 Setting up Compel for long prompts...")
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False
            )
            
            # Setup Compel for OpenPose pipeline if available
            if self.openpose_pipeline:
                print("   🚀 Setting up Compel for OpenPose pipeline...")
                self.openpose_compel = Compel(
                    tokenizer=[self.openpose_pipeline.tokenizer, self.openpose_pipeline.tokenizer_2],
                    text_encoder=[self.openpose_pipeline.text_encoder, self.openpose_pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False
                )
            
            print("✅ Pipeline with Compel ready!")
            print("   🎯 Long prompts (>77 tokens) now supported!")
            if self.openpose_pipeline:
                print("   🦴 OpenPose comparison enabled!")
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
            
            # Load IP-Adapter for OpenPose pipeline if available
            if self.openpose_pipeline:
                print("   🖼️ Loading IP-Adapter for OpenPose pipeline...")
                self.openpose_pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
                )
                self.openpose_ip_adapter_loaded = True
                print("   ✅ IP-Adapter loaded for OpenPose pipeline!")
            
            return True
            
        except Exception as e:
            print(f"   ❌ IP-Adapter failed: {e}")
            return False
    
    def create_openpose_image(self, scene_type, width=1024, height=1024):
        """Create OpenPose skeleton for the scene"""
        img = Image.new('RGB', (width, height), 'black')
        draw = ImageDraw.Draw(img)
        
        # OpenPose colors
        colors = {
            'nose': (255, 0, 0),
            'neck': (255, 85, 0),
            'shoulder': (255, 170, 0),
            'elbow': (255, 255, 0),
            'wrist': (170, 255, 0),
            'hip': (85, 255, 0),
            'knee': (0, 255, 0),
            'ankle': (0, 255, 85),
            'body': (0, 170, 255)
        }
        
        line_width = 6
        joint_radius = 8
        
        if scene_type == "single":
            # Single character in center
            self._draw_openpose_figure(draw, width//2, height//2, height//3, 
                                     colors, line_width, joint_radius)
        else:  # dual
            # Two characters side by side - SAME HEIGHT
            self._draw_openpose_figure(draw, width//3, height//2, height//3, 
                                     colors, line_width, joint_radius)
            self._draw_openpose_figure(draw, 2*width//3, height//2, height//3, 
                                     colors, line_width, joint_radius)
        
        return img
    
    def _draw_openpose_figure(self, draw, center_x, center_y, height, colors, line_width, joint_radius):
        """Draw a single OpenPose figure"""
        # Calculate proportions
        head_size = height * 0.12
        neck_height = height * 0.1
        torso_height = height * 0.25
        upper_leg = height * 0.2
        lower_leg = height * 0.2
        upper_arm = height * 0.15
        lower_arm = height * 0.15
        shoulder_width = height * 0.2
        hip_width = height * 0.15
        
        # Define keypoints
        keypoints = {
            'nose': (center_x, center_y - height//2 + head_size//2),
            'neck': (center_x, center_y - height//2 + head_size + neck_height),
            'right_shoulder': (center_x - shoulder_width//2, center_y - height//2 + head_size + neck_height),
            'left_shoulder': (center_x + shoulder_width//2, center_y - height//2 + head_size + neck_height),
            'right_hip': (center_x - hip_width//2, center_y - height//2 + head_size + neck_height + torso_height),
            'left_hip': (center_x + hip_width//2, center_y - height//2 + head_size + neck_height + torso_height),
            'right_elbow': (center_x - shoulder_width//2 - upper_arm*0.7, center_y - height//2 + head_size + neck_height + upper_arm),
            'left_elbow': (center_x + shoulder_width//2 + upper_arm*0.7, center_y - height//2 + head_size + neck_height + upper_arm),
            'right_wrist': (center_x - shoulder_width//2 - upper_arm*0.7, center_y - height//2 + head_size + neck_height + upper_arm + lower_arm),
            'left_wrist': (center_x + shoulder_width//2 + upper_arm*0.7, center_y - height//2 + head_size + neck_height + upper_arm + lower_arm),
            'right_knee': (center_x - hip_width//2, center_y - height//2 + head_size + neck_height + torso_height + upper_leg),
            'left_knee': (center_x + hip_width//2, center_y - height//2 + head_size + neck_height + torso_height + upper_leg),
            'right_ankle': (center_x - hip_width//2, center_y - height//2 + head_size + neck_height + torso_height + upper_leg + lower_leg),
            'left_ankle': (center_x + hip_width//2, center_y - height//2 + head_size + neck_height + torso_height + upper_leg + lower_leg),
        }
        
        # Define connections
        connections = [
            ('nose', 'neck', colors['neck']),
            ('neck', 'right_shoulder', colors['shoulder']),
            ('neck', 'left_shoulder', colors['shoulder']),
            ('right_shoulder', 'right_elbow', colors['elbow']),
            ('right_elbow', 'right_wrist', colors['wrist']),
            ('left_shoulder', 'left_elbow', colors['elbow']),
            ('left_elbow', 'left_wrist', colors['wrist']),
            ('neck', 'right_hip', colors['body']),
            ('neck', 'left_hip', colors['body']),
            ('right_hip', 'left_hip', colors['hip']),
            ('right_hip', 'right_knee', colors['knee']),
            ('right_knee', 'right_ankle', colors['ankle']),
            ('left_hip', 'left_knee', colors['knee']),
            ('left_knee', 'left_ankle', colors['ankle']),
        ]
        
        # Draw connections
        for start, end, color in connections:
            if start in keypoints and end in keypoints:
                x1, y1 = keypoints[start]
                x2, y2 = keypoints[end]
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        
        # Draw joints
        for joint_name, (x, y) in keypoints.items():
            if 'nose' in joint_name:
                color = colors['nose']
            elif 'shoulder' in joint_name:
                color = colors['shoulder']
            elif 'elbow' in joint_name:
                color = colors['elbow']
            elif 'wrist' in joint_name:
                color = colors['wrist']
            elif 'hip' in joint_name:
                color = colors['hip']
            elif 'knee' in joint_name:
                color = colors['knee']
            elif 'ankle' in joint_name:
                color = colors['ankle']
            else:
                color = colors['neck']
                
            draw.ellipse([(x-joint_radius, y-joint_radius), 
                          (x+joint_radius, y+joint_radius)], 
                         fill=color, outline='white', width=2)
    
    def encode_prompts_with_compel(self, positive_prompt, negative_prompt, use_openpose_compel=False):
        """Encode both prompts using Compel and ensure matching shapes"""
        try:
            # Choose the right Compel instance
            compel_instance = self.openpose_compel if use_openpose_compel and self.openpose_compel else self.compel
            
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
                                  use_openpose=False, openpose_image=None, **kwargs):
        """Generate scene with or without OpenPose"""
        
        if use_openpose and not self.openpose_pipeline:
            print("   ❌ OpenPose pipeline not available!")
            return None
        
        try:
            # Choose pipeline and compel
            if use_openpose:
                pipeline = self.openpose_pipeline
                compel_for_encoding = True  # Use openpose compel
                if not self.openpose_ip_adapter_loaded:
                    print("   ❌ IP-Adapter not loaded for OpenPose!")
                    return None
            else:
                pipeline = self.pipeline
                compel_for_encoding = False  # Use regular compel
                if not self.ip_adapter_loaded:
                    print("   ❌ IP-Adapter not loaded!")
                    return None
            
            # Encode prompts
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt, use_openpose_compel=compel_for_encoding
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
                
                if use_openpose and openpose_image:
                    generation_kwargs['image'] = openpose_image
                    generation_kwargs['controlnet_conditioning_scale'] = 1.0
                
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
                
                if use_openpose and openpose_image:
                    generation_kwargs['image'] = openpose_image
                    generation_kwargs['controlnet_conditioning_scale'] = 1.0
            
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
    """RealCartoon XL v7 with Compel + OpenPose Comparison"""
    print("\n" + "="*70)
    print("🚀 REALCARTOON XL V7 WITH COMPEL + OPENPOSE COMPARISON")
    print("   Generate scenes WITH and WITHOUT OpenPose for dissertation")
    print("="*70)
    
    print("\n🎯 Features:")
    print("  • Handles prompts longer than 77 tokens (Compel)")
    print("  • Generates BOTH versions for comparison")
    print("  • WITHOUT OpenPose: IP-Adapter only")
    print("  • WITH OpenPose: IP-Adapter + skeleton control")
    print("  • Perfect for dissertation comparison!")
    
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
        ("01_alex_standing", "single", "Alex", 
         "Alex standing alone in the classroom, full body visible, friendly smile"),
        
        ("02_emma_standing", "single", "Emma", 
         "Emma standing alone in the classroom, full body visible, gentle expression"),
        
        # Dual character scenes - these will show height differences
        ("03_standing_together", "dual", "both", 
         "Alex and Emma standing side by side, both children the same height, full bodies visible"),
        
        ("04_high_five", "dual", "both", 
         "Alex and Emma doing a high five together, hands raised up and touching, both smiling"),
        
        ("05_sitting_together", "dual", "both", 
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
        
        # PHASE 3: Generate scenes WITH and WITHOUT OpenPose
        print(f"\n🎬 PHASE 3: Comparison Scenes ({len(educational_scenes)} total)")
        print("   Each scene will be generated TWICE:")
        print("   • WITHOUT OpenPose (IP-Adapter only)")
        print("   • WITH OpenPose (IP-Adapter + skeleton control)")
        print("-" * 70)
        
        results = []
        total_time = 0
        
        for i, (scene_name, scene_type, character_focus, action) in enumerate(educational_scenes):
            print(f"\n🎨 Scene {i+1}: {scene_name} ({scene_type})")
            
            prompt = prompts.get_scene_prompt("social skills", character_focus, action)
            print(f"   📊 Prompt length: {len(prompt.split())} words (no truncation!)")
            
            # Prepare character refs
            if scene_type == "single":
                character_refs = [alex_ref if character_focus == "Alex" else emma_ref]
            else:
                character_refs = [alex_ref, emma_ref]
            
            # Generate OpenPose skeleton if available
            openpose_image = None
            if pipeline.openpose_pipeline:
                openpose_image = pipeline.create_openpose_image(scene_type)
                openpose_path = os.path.join(output_dir, f"{scene_name}_openpose_skeleton.png")
                openpose_image.save(openpose_path)
                print(f"   💀 Saved OpenPose skeleton: {openpose_path}")
            
            # Generate WITHOUT OpenPose
            print(f"   🖼️ Generating WITHOUT OpenPose...")
            start_time = time.time()
            
            image_without = pipeline.generate_scene_with_compel(
                prompt, negative, scene_type, character_refs,
                use_openpose=False,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=1024, width=1024,
                generator=torch.Generator(pipeline.device).manual_seed(3000 + i)
            )
            
            without_time = time.time() - start_time
            
            if image_without:
                without_path = os.path.join(output_dir, f"{scene_name}_WITHOUT_openpose.png")
                image_without.save(without_path, quality=95, optimize=True)
                print(f"   ✅ WITHOUT OpenPose: {without_time:.1f}s")
            
            # Generate WITH OpenPose if available
            if pipeline.openpose_pipeline and openpose_image:
                print(f"   🖼️ Generating WITH OpenPose...")
                start_time = time.time()
                
                image_with = pipeline.generate_scene_with_compel(
                    prompt, negative, scene_type, character_refs,
                    use_openpose=True,
                    openpose_image=openpose_image,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(3000 + i)
                )
                
                with_time = time.time() - start_time
                total_time += without_time + with_time
                
                if image_with:
                    with_path = os.path.join(output_dir, f"{scene_name}_WITH_openpose.png")
                    image_with.save(with_path, quality=95, optimize=True)
                    print(f"   ✅ WITH OpenPose: {with_time:.1f}s")
                    results.append(scene_name)
            else:
                total_time += without_time
                print("   ⚠️ Skipping OpenPose version (not available)")
        
        # Results
        print("\n" + "=" * 70)
        print("🚀 OPENPOSE COMPARISON RESULTS")
        print("=" * 70)
        
        print(f"✓ Generated {len(results)} complete scene comparisons")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Used Compel for long prompts (no truncation!)")
        
        print("\n📋 Generated files:")
        print(f"   📁 {output_dir}/")
        print("   • alex_reference.png & emma_reference.png")
        for scene_name in results:
            print(f"   • {scene_name}_WITHOUT_openpose.png")
            print(f"   • {scene_name}_WITH_openpose.png")
            print(f"   • {scene_name}_openpose_skeleton.png")
        
        print("\n🎓 FOR YOUR DISSERTATION:")
        print("   • Compare heights in dual character scenes")
        print("   • WITHOUT: Heights may be inconsistent")
        print("   • WITH: OpenPose enforces skeleton heights")
        print("   • Same height skeletons = same height characters")
        print("   • Clear visual evidence of control benefits")
        
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
        if pipeline.openpose_pipeline:
            del pipeline.openpose_pipeline
        if pipeline.compel:
            del pipeline.compel
        if pipeline.openpose_compel:
            del pipeline.openpose_compel
        if pipeline.openpose_controlnet:
            del pipeline.openpose_controlnet
        torch.cuda.empty_cache()
        print("✅ Complete")


if __name__ == "__main__":
    main()