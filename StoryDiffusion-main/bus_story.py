import os
import torch
import time
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from diffusers import StableDiffusionXLControlNetPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, ControlNetModel
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection, CLIPProcessor, CLIPModel
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc
import numpy as np
from typing import List, Dict, Optional, Tuple
import hashlib

class SingleReferenceControlNetPipeline:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.compel = None
        self.ip_adapter_loaded = False
        self.face_ip_adapter_loaded = False
        self.controlnet = None
        
        self.master_seed = 42
        self.scene_seeds = self._generate_fixed_scene_seeds()
        self.reference_seed = 1000
        self.driver_reference_seed = 2000
        
        self.adam_reference = None
        self.adam_face_reference = None
        self.driver_reference = None
        
        self.consistency_scorer = None
        
        # Initialize mask processor
        self.mask_processor = IPAdapterMaskProcessor()
        
    def _generate_fixed_scene_seeds(self):
        scenes = [
            "reference_adam",
            "reference_driver",
            "01_buying_ticket",
            "02_walking_to_stop",
            "03_waiting_at_stop",
            "04_showing_ticket",
            "05_taking_seat"
        ]
        
        seeds = {}
        base_seed = self.master_seed
        
        for i, scene in enumerate(scenes):
            scene_hash = int(hashlib.md5(scene.encode()).hexdigest()[:8], 16)
            seeds[scene] = (base_seed + scene_hash) % 100000
            
            if scene == "04_showing_ticket":
                seeds[scene] = (seeds[scene] + 7777) % 100000
            elif scene == "01_buying_ticket":
                seeds[scene] = (seeds[scene] + 3333) % 100000
            elif scene == "reference_driver":
                seeds[scene] = (seeds[scene] + 5555) % 100000
                
        return seeds
    
    def setup_pipeline_with_full_ip_adapter(self):
        try:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                torch_dtype=torch.float16,
            )
            
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0-small",
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16,
            )
            
            self.pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                self.model_path,
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                image_encoder=image_encoder,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config
            )
            self.pipeline.enable_vae_tiling()
            self.pipeline.enable_model_cpu_offload()
            
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False
            )
            
            self.setup_consistency_scorer()
            
            return True
            
        except Exception as e:
            traceback.print_exc()
            return False
    
    def setup_consistency_scorer(self):
        try:
            self.consistency_scorer = {
                'model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                'processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            }
        except Exception as e:
            pass
    
    def load_full_ip_adapter(self):
        try:
            # Load multiple IP adapters for both full and face
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=[
                    "ip-adapter-plus_sdxl_vit-h.safetensors",
                    "ip-adapter-plus-face_sdxl_vit-h.safetensors"
                ]
            )
            self.ip_adapter_loaded = True
            self.face_ip_adapter_loaded = True
            return True
            
        except Exception:
            # Fallback to single IP adapter if multiple loading fails
            try:
                self.pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
                )
                self.ip_adapter_loaded = True
                self.face_ip_adapter_loaded = False
                print("Warning: Face IP adapter not loaded, using only full IP adapter")
                return True
            except Exception:
                return False
    
    def create_ip_adapter_mask(self, width=1024, height=1024, scene_type="single"):
        """Create simple 50/50 split masks with blur and gap for better blending"""
        if scene_type == "dual_scene4":
            mask_adam = Image.new('L', (width, height), 0)
            mask_driver = Image.new('L', (width, height), 0)
            
            draw_adam = ImageDraw.Draw(mask_adam)
            draw_driver = ImageDraw.Draw(mask_driver)
            
            gap_size = 40
            
            driver_right = (width // 2) - (gap_size // 2)
            draw_driver.rectangle([0, 0, driver_right, height], fill=255)
            
            adam_left = (width // 2) + (gap_size // 2)
            draw_adam.rectangle([adam_left, 0, width, height], fill=255)
            
            blur_radius = 8
            mask_driver = mask_driver.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask_adam = mask_adam.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            try:
                mask_driver.save("debug_driver_mask.png")
                mask_adam.save("debug_adam_mask.png")
                print("Saved 50/50 split masks with gap and blur")
            except:
                pass
            
            return [mask_adam, mask_driver]
        
        else:
            return [Image.new('L', (width, height), 255)]
    
    def prepare_face_reference(self, image):
        """Prepare the reference image for face IP adapter (no cropping)"""
        # Use the full image for face IP adapter as well
        return image
    
    def create_reference_image(self, reference_prompt, negative_prompt, character_type="adam"):
        try:
            if character_type == "adam":
                reference_seed = self.scene_seeds.get("reference_adam", self.reference_seed)
            else:  # driver
                reference_seed = self.scene_seeds.get("reference_driver", self.driver_reference_seed)
            
            reference_image = self._generate_single_reference(
                reference_prompt, negative_prompt, reference_seed
            )
            
            # Prepare face reference for Adam (using full image)
            if character_type == "adam" and reference_image:
                self.adam_face_reference = self.prepare_face_reference(reference_image)
            
            return reference_image
            
        except Exception as e:
            return None
    
    def _generate_single_reference(self, prompt, negative_prompt, seed):
        try:
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = \
                self.encode_prompts_with_compel(prompt, negative_prompt)
            
            if prompt_embeds is None:
                return None
            
            depth_map = self.create_adaptive_depth_map("full_body", 1024, 1024)
            
            params = {
                "num_inference_steps": 30,
                "guidance_scale": 9.0,
                "height": 1024,
                "width": 1024,
                "generator": torch.Generator(self.device).manual_seed(seed)
            }
            
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                image=depth_map,
                controlnet_conditioning_scale=0.2,
                **params
            )
            
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            return None
    
    def encode_prompts_with_compel(self, positive_prompt, negative_prompt):
        try:
            positive_conditioning, positive_pooled = self.compel(positive_prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)
            
            [positive_conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
            return positive_conditioning, positive_pooled, negative_conditioning, negative_pooled
            
        except Exception as e:
            return None, None, None, None
    
    def create_adaptive_depth_map(self, scene_type="portrait", width=1024, height=1024):
        if scene_type == "portrait":
            depth_map = np.ones((height, width), dtype=np.uint8) * 128
            center_x, center_y = width // 2, height // 3
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            max_dist = np.sqrt(center_x**2 + center_y**2)
            depth_gradient = 1 - (dist_from_center / max_dist)
            depth_map = (128 + depth_gradient * 50).astype(np.uint8)
            
        elif scene_type == "full_body":
            depth_map = np.linspace(100, 156, height)[:, np.newaxis]
            depth_map = np.repeat(depth_map, width, axis=1).astype(np.uint8)
            
        elif scene_type == "medium_shot":
            depth_map = np.ones((height, width), dtype=np.uint8) * 140
            center_x, center_y = width // 2, height // 3.5
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            max_dist = np.sqrt(center_x**2 + center_y**2) * 0.7
            depth_gradient = 1 - (dist_from_center / max_dist)
            depth_map = (140 + depth_gradient * 40).astype(np.uint8)
            
        elif scene_type == "close_up":
            depth_map = np.ones((height, width), dtype=np.uint8) * 150
            center_x, center_y = width // 2, height // 2.5
            Y, X = np.ogrid[:height, :width]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            max_dist = np.sqrt(center_x**2 + center_y**2) * 0.5
            depth_gradient = 1 - (dist_from_center / max_dist)
            depth_map = (150 + depth_gradient * 30).astype(np.uint8)
            
        elif scene_type == "bus_interior":
            # Special depth map for bus interior scene
            depth_map = np.ones((height, width), dtype=np.uint8) * 135
            
            # Driver area (left side) - closer to camera
            depth_map[:, :width//2] = 145
            
            # Passenger area (right side) - slightly further
            depth_map[:, width//2:] = 125
            
            # Add some depth variation
            Y, X = np.ogrid[:height, :width]
            depth_variation = np.sin(X / width * np.pi) * 10
            depth_map = np.clip(depth_map + depth_variation, 100, 200).astype(np.uint8)
            
        else:
            depth_map = np.ones((height, width), dtype=np.uint8) * 128
        
        return Image.fromarray(depth_map, mode='L')
    
    def _generate_dual_character_scene4(self, scene_name, prompt, negative_prompt, seed, **kwargs):
        """Generate scene 4 with both Adam and bus driver using masked IP adapter"""
        try:
            print("Generating scene 4 with both characters using proper masking...")
            
            # For dual character scene, we'll use single IP adapter with both references
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"]
            )
            
            # Set scale for multiple images
            self.pipeline.set_ip_adapter_scale([[0.8, 0.3]])  # [Adam scale, Driver scale]
            
            # Create and preprocess masks
            raw_masks = self.create_ip_adapter_mask(scene_type="dual_scene4")
            
            # Preprocess masks using IPAdapterMaskProcessor
            processed_masks = self.mask_processor.preprocess(
                raw_masks,
                height=1024,
                width=1024
            )
            
            # Prepare images as nested list
            ip_images = [[self.adam_reference, self.driver_reference]]
            
            # Reshape masks correctly for pipeline
            masks = [processed_masks.reshape(1, processed_masks.shape[0], processed_masks.shape[2], processed_masks.shape[3])]
            
            # Generate with proper parameters
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = \
                self.encode_prompts_with_compel(prompt, negative_prompt)
            
            depth_map = self.create_adaptive_depth_map("bus_interior")
            
            gen_params = kwargs.copy()
            gen_params['generator'] = torch.Generator(self.device).manual_seed(seed)
            gen_params.pop('seed', None)
            gen_params['guidance_scale'] = 12.0
            gen_params['num_inference_steps'] = 30
            
            print(f"Using IP adapter images: {len(ip_images[0])} images")
            print(f"Using masks shape: {masks[0].shape}")
            
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                ip_adapter_image=ip_images,
                image=depth_map,
                controlnet_conditioning_scale=0.1,
                cross_attention_kwargs={"ip_adapter_masks": masks},
                **gen_params
            )
            
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"Dual character generation failed: {e}")
            traceback.print_exc()
            return None
    
    def generate_scene_with_reference(self, scene_name, prompt, negative_prompt, **kwargs):
        try:
            fixed_seed = self.scene_seeds.get(scene_name, self.master_seed)
            
            # Special handling for scene 4 with dual characters
            if scene_name == "04_showing_ticket" and self.ip_adapter_loaded and self.adam_reference and self.driver_reference:
                print("Attempting dual character scene 4...")
                result = self._generate_dual_character_scene4(scene_name, prompt, negative_prompt, fixed_seed, **kwargs)
                if result:
                    # Reload the dual IP adapters after scene 4
                    print("Reloading IP adapters for subsequent scenes...")
                    self.load_full_ip_adapter()
                    return result
                print("Dual character failed, using fallback...")
            
            # Regular single character generation with both full and face IP adapters
            if self.face_ip_adapter_loaded and self.adam_face_reference:
                # Use both full and face IP adapters for Adam
                reference_images = [self.adam_reference, self.adam_face_reference]
                
                # Set different scales for full and face IP adapters
                if scene_name in ["04_showing_ticket"]:
                    self.pipeline.set_ip_adapter_scale([0.20, 0.1])  # [full, face]
                elif scene_name in ["01_buying_ticket", "02_walking_to_stop", "03_waiting_at_stop", "05_taking_seat"]:
                    self.pipeline.set_ip_adapter_scale([0.25, 0.1])  # [full, face]
                else:
                    self.pipeline.set_ip_adapter_scale([0.25, 0.1])  # [full, face]
            else:
                # Fallback to single IP adapter
                reference_images = self.adam_reference
                
                if scene_name in ["04_showing_ticket"]:
                    self.pipeline.set_ip_adapter_scale(0.20)
                elif scene_name in ["01_buying_ticket", "02_walking_to_stop", "03_waiting_at_stop", "05_taking_seat"]:
                    self.pipeline.set_ip_adapter_scale(0.35)
                else:
                    self.pipeline.set_ip_adapter_scale(0.25)
            
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = \
                self.encode_prompts_with_compel(prompt, negative_prompt)
            
            if prompt_embeds is None:
                return None
            
            # Determine scene type
            if "medium shot" in prompt.lower() or scene_name in ["01_buying_ticket", "04_showing_ticket"]:
                scene_type = "medium_shot"
            elif "close-up" in prompt.lower() or "close up" in prompt.lower():
                scene_type = "close_up"
            elif "full body" in prompt.lower():
                scene_type = "full_body"
            else:
                scene_type = "portrait"
            
            depth_map = self.create_adaptive_depth_map(scene_type)
            
            gen_params = kwargs.copy()
            gen_params['generator'] = torch.Generator(self.device).manual_seed(fixed_seed)
            gen_params.pop('seed', None)
            
            # Higher guidance to force prompt adherence
            if scene_name in ["04_showing_ticket"]:
                gen_params['guidance_scale'] = 15.0  
            elif scene_name in ["01_buying_ticket"]:
                gen_params['guidance_scale'] = 14.0
            elif scene_name in ["02_walking_to_stop", "03_waiting_at_stop"]:
                gen_params['guidance_scale'] = 12.0
            else:
                gen_params['guidance_scale'] = 10.0
            
            if self.ip_adapter_loaded and reference_images is not None:
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    ip_adapter_image=reference_images,
                    image=depth_map,
                    controlnet_conditioning_scale=0.1,
                    **gen_params
                )
            else:
                result = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    negative_pooled_prompt_embeds=negative_pooled,
                    image=depth_map,
                    controlnet_conditioning_scale=0.1,
                    **gen_params
                )
            
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            traceback.print_exc()
            return None

class EnhancedClothingPrompts:
    def __init__(self):
        # Simplified consistent character description
        self.adam_base = "young man Adam with brown hair"
        self.driver_base = "middle-aged bus driver with gray beard, professional appearance"
        
        # Simplified outfit that's easy to maintain
        self.adam_outfit = "shirt, blue jacket, no buttons, jeans"
        self.driver_outfit = "navy blue bus driver uniform, cap, name badge"
        
        self.style = "cartoon illustration"
        
        # SHORTENED prompts to avoid token limit issues
        self.scene_prompts = {
            "01_buying_ticket": (
                f"man pressing buttons on ticket machine, finger touching machine"
                f"{self.adam_base}, {self.adam_outfit}, detailed ticket machine, {self.style}"
            ),
            
            "02_walking_to_stop": (
                f"man walking on sidewalk"
                f"{self.adam_base}, {self.adam_outfit}, full body, {self.style}"
            ),
            
            "03_waiting_at_stop": (
                f"man standing at bus stop, bus stop sign visible, no text, no words"
                f"{self.adam_base}, {self.adam_outfit}, {self.style}"
            ),
            
            "04_showing_ticket": (
                "bus interior scene, passenger Adam on right side, Adam wearing gray shirt"
                "bus driver on left side taking ticket from passenger"
                "driver seated behind wheel, hand-to-hand ticket exchange, cartoon illustration"
            ),
            
            "05_taking_seat": (
                f"sitting down in bus seat "
                f"{self.adam_base}, {self.adam_outfit}, {self.style}"
            )
        }
        
        self.base_negative = (
            "realistic photo, wrong character, inconsistent appearance, "
            "blurry face, deformed, extra limbs, bad anatomy"
        )
        
        # Outfit-specific negative prompts to enforce consistency
        self.adam_outfit_negative = (
            "different clothing, red shirt, green shirt, no jacket, shorts, pants other than jeans, "
            "hat, glasses, accessories, mismatched outfit"
        )
        
        self.driver_outfit_negative = (
            "casual clothes, no uniform, wrong uniform color, no cap, civilian clothes, "
            "jacket other than uniform, mismatched driver outfit"
        )
        
        # Much more specific negative prompts
        self.scene_negatives = {
            "01_buying_ticket": (
                "no machine, no pressing, no interaction, empty hands, no buttons, "
                "sitting, lying down, walking away, text, letters, blue eyes"
            ),
            "02_walking_to_stop": (
                "complex scenery, standing still, no ticket visible, empty hands, no walking, sitting, "
                "no movement, static pose, artifacts, multiple people, blue eyes, complex background, large ticket, children, duplicates, complex background, unclear face"
            ),
            "03_waiting_at_stop": (
                "no bus stop,, words, letters, walking, sitting, "
                "no standing, missing bus stop sign, letters, text"
            ),
            "04_showing_ticket": (
                "two tickets, blurry, smudges, artifacts, two papers, no hand extended, no ticket visible, no driver, no exchange, empty hands, "
                "hands hidden, no interaction, sitting, walking away, no uniform, "
                "no steering wheel, outside bus, missing driver, no bus interior, "
                "only one person, missing passenger, missing driver, single character, artifact"
            ),
            "05_taking_seat": (
                "standing, no seat, outside bus, no sitting, empty bus, "
                "no interior, walking, no bus seat visible, blue eyes"
            )
        }
    
    def get_reference_prompt(self, character_type="adam"):
        if character_type == "adam":
            return f"{self.adam_base}, {self.adam_outfit}, full body portrait, white background, {self.style}"
        else:  # driver
            return f"{self.driver_base}, {self.driver_outfit}, full body portrait, white background, {self.style}"
    
    def get_scene_prompt(self, scene_name):
        return self.scene_prompts.get(scene_name, f"{self.adam_base}, {self.style}")
    
    def get_negative_prompt(self, scene_name=None, character_type="adam"):
        # Combine base negative, outfit negative, and scene-specific negative
        negatives = [self.base_negative]
        
        if character_type == "adam":
            negatives.append(self.adam_outfit_negative)
        else:  # driver
            negatives.append(self.driver_outfit_negative)
            
        if scene_name and scene_name in self.scene_negatives:
            negatives.append(self.scene_negatives[scene_name])
        return ", ".join(negatives)

def find_realcartoon_model():
    paths = [
        "../models/realcartoonxl_v7.safetensors",
        "models/realcartoonxl_v7.safetensors",
        "../models/RealCartoon-XL-v7.safetensors",
        "realcartoonxl_v7.safetensors",
        "RealCartoon-XL-v7.safetensors"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def generate_scene_with_reference(pipeline, prompts, scene_name, output_dir):
    prompt = prompts.get_scene_prompt(scene_name)
    negative = prompts.get_negative_prompt(scene_name)
    
    params = {
        "num_inference_steps": 30,
        "guidance_scale": 8.0,
        "height": 1024,
        "width": 1024
    }
    
    print(f"\nGenerating {scene_name}...")
    
    image = pipeline.generate_scene_with_reference(
        scene_name, prompt, negative, **params
    )
    
    if image:
        output_path = os.path.join(output_dir, f"{scene_name}.png")
        image.save(output_path, quality=95, optimize=True)
        print(f"Saved: {output_path}")
        return True
    else:
        print(f"Failed to generate {scene_name}")
        return False

def create_bus_journey_storyboard(output_dir, include_references=True):
    """Create a storyboard combining all bus journey images with labels for autism support"""
    
    print("\n📋 Creating Bus Journey Storyboard...")
    print("-" * 50)
    
    # Define scene descriptions for autism support
    scene_descriptions = {
        "adam_reference": "This is Adam",
        "driver_reference": "This is the Bus Driver",
        "01_buying_ticket": "Step 1: Adam buys a bus ticket",
        "02_walking_to_stop": "Step 2: Adam walks to the bus stop",
        "03_waiting_at_stop": "Step 3: Adam waits at the bus stop",
        "04_showing_ticket": "Step 4: Adam shows the driver his ticket",
        "05_taking_seat": "Step 5: Adam sits down on the bus"
    }
    
    # Collect images
    images = []
    labels = []
    
    # Add Adam reference image only (not driver reference)
    if include_references:
        adam_ref_path = os.path.join(output_dir, "adam_reference.png")
        if os.path.exists(adam_ref_path):
            images.append(Image.open(adam_ref_path))
            labels.append(scene_descriptions.get("adam_reference", "adam_reference"))
            print(f"  ✅ Added: {labels[-1]}")
    
    # Add scene images
    scenes = [
        "01_buying_ticket",
        "02_walking_to_stop",
        "03_waiting_at_stop",
        "04_showing_ticket",
        "05_taking_seat"
    ]
    
    for scene_name in scenes:
        scene_path = os.path.join(output_dir, f"{scene_name}.png")
        if os.path.exists(scene_path):
            images.append(Image.open(scene_path))
            labels.append(scene_descriptions.get(scene_name, scene_name))
            print(f"  ✅ Added: {labels[-1]}")
    
    if not images:
        print("  ❌ No images found for storyboard")
        return None
    
    # Configuration similar to Alex version - compact but with text wrapping
    img_size = 320
    padding_horizontal = 40  # Reduced padding
    padding_vertical = 30    # Reduced padding
    label_spacing = 10       # Space between image and label
    title_height = 80
    footer_height = 60
    
    # Calculate grid layout
    num_images = len(images)
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    # Try to load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        footer_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        try:
            title_font = ImageFont.truetype("arial.ttf", 36)
            label_font = ImageFont.truetype("arial.ttf", 18)
            footer_font = ImageFont.truetype("arial.ttf", 18)
        except:
            title_font = ImageFont.load_default()
            label_font = title_font
            footer_font = title_font
    
    # Function to wrap text to fit within card width
    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        
        temp_img = Image.new('RGB', (100, 100))
        temp_draw = ImageDraw.Draw(temp_img)
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)  # Single word longer than max_width
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    # Calculate maximum label height needed for wrapped text
    card_padding = 15
    text_area_width = img_size  # Text area same width as image
    max_label_height = 0
    
    for label in labels:
        wrapped_lines = wrap_text(label, label_font, text_area_width)
        line_height = 22  # Height per line of text
        total_text_height = len(wrapped_lines) * line_height
        max_label_height = max(max_label_height, total_text_height)
    
    # Calculate canvas size
    card_width = img_size + 2 * card_padding
    card_height = img_size + max_label_height + label_spacing + 2 * card_padding
    
    canvas_width = cols * card_width + (cols + 1) * padding_horizontal
    canvas_height = title_height + rows * card_height + (rows + 1) * padding_vertical + footer_height
    
    # Create canvas with light background
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='#F5F5F5')
    draw = ImageDraw.Draw(canvas)
    
    # Add title with background
    title = "Adam's Bus Journey"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    title_y = 25
    
    # Draw title background
    draw.rectangle([0, 0, canvas_width, title_height], fill='#2196F3')
    draw.text((title_x, title_y), title, fill='white', font=title_font)
    
    # Place images in grid
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = padding_horizontal + col * (card_width + padding_horizontal)
        y = title_height + padding_vertical + row * (card_height + padding_vertical)
        
        # Draw shadow effect
        shadow_offset = 3
        draw.rectangle([x + shadow_offset, y + shadow_offset, 
                       x + card_width + shadow_offset, y + card_height + shadow_offset], 
                      fill='#E0E0E0', outline=None)
        
        # Draw card
        draw.rectangle([x, y, x + card_width, y + card_height], 
                      fill='white', outline='#E0E0E0', width=2)
        
        # Resize and paste image
        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img_x = x + card_padding
        img_y = y + card_padding
        canvas.paste(img_resized, (img_x, img_y))
        
        # Draw wrapped text
        text_start_y = y + card_padding + img_size + label_spacing
        wrapped_lines = wrap_text(label, label_font, text_area_width)
        
        for line_idx, line in enumerate(wrapped_lines):
            # Center each line
            line_bbox = draw.textbbox((0, 0), line, font=label_font)
            line_width = line_bbox[2] - line_bbox[0]
            line_x = x + card_padding + (img_size - line_width) // 2
            line_y = text_start_y + line_idx * 22
            
            draw.text((line_x, line_y), line, fill='#333333', font=label_font)
    
    # Add footer with background
    footer_y_start = canvas_height - footer_height
    draw.rectangle([0, footer_y_start, canvas_width, canvas_height], fill='#E3F2FD')
    
    footer_text = "This storyboard shows Adam taking the bus step by step"
    footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
    footer_width = footer_bbox[2] - footer_bbox[0]
    footer_x = (canvas_width - footer_width) // 2
    footer_y = footer_y_start + (footer_height - (footer_bbox[3] - footer_bbox[1])) // 2
    
    draw.text((footer_x, footer_y), footer_text, fill='#555555', font=footer_font)
    
    # Save storyboard
    storyboard_path = os.path.join(output_dir, "adam_bus_journey_storyboard.png")
    canvas.save(storyboard_path, quality=95, optimize=True)
    
    print(f"\n✅ Storyboard saved: {storyboard_path}")
    print(f"  📐 Size: {canvas_width}x{canvas_height}px")
    print(f"  📚 Contains {len(images)} images with descriptive labels")
    print(f"  🎯 Designed for autism support with clear visual steps")
    print(f"  🎨 Features: Card design, shadows, numbered steps, text wrapping")
    
    return storyboard_path

def main():
    model_path = find_realcartoon_model()
    if not model_path:
        model_path = input("Enter RealCartoon XL v7 path: ").strip()
        if not os.path.exists(model_path):
            print("Model not found!")
            return
    
    output_dir = "adam_bus_journey_consistent"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Setting up pipeline...")
        pipeline = SingleReferenceControlNetPipeline(model_path)
        if not pipeline.setup_pipeline_with_full_ip_adapter():
            print("Failed to setup pipeline!")
            return
        
        prompts = EnhancedClothingPrompts()
        
        # Generate Adam reference image
        print("\nGenerating Adam reference image...")
        adam_reference_prompt = prompts.get_reference_prompt("adam")
        adam_negative = prompts.get_negative_prompt(character_type="adam")
        
        adam_ref = pipeline.create_reference_image(
            adam_reference_prompt, adam_negative, "adam"
        )
        
        if not adam_ref:
            print("Failed to create Adam reference image!")
            return
        
        adam_ref.save(os.path.join(output_dir, "adam_reference.png"))
        print("Adam reference image saved!")
        
        # Note: Face reference is the same as full reference (no cropping)
        if pipeline.adam_face_reference:
            print("Adam face reference prepared (using full image)!")
        
        pipeline.adam_reference = adam_ref
        
        # Generate Bus Driver reference image
        print("\nGenerating Bus Driver reference image...")
        driver_reference_prompt = prompts.get_reference_prompt("driver")
        driver_negative = prompts.get_negative_prompt(character_type="driver")
        
        driver_ref = pipeline.create_reference_image(
            driver_reference_prompt, driver_negative, "driver"
        )
        
        if not driver_ref:
            print("Failed to create Driver reference image!")
            return
        
        driver_ref.save(os.path.join(output_dir, "driver_reference.png"))
        print("Driver reference image saved!")
        
        pipeline.driver_reference = driver_ref
        
        # Load IP adapter
        print("\nLoading IP adapter...")
        if not pipeline.load_full_ip_adapter():
            print("Warning: Failed to load IP adapter, continuing without it...")
        
        # Generate scenes
        scenes = [
            "01_buying_ticket",
            "02_walking_to_stop",
            "03_waiting_at_stop",
            "04_showing_ticket",
            "05_taking_seat"
        ]
        
        for scene_name in scenes:
            generate_scene_with_reference(pipeline, prompts, scene_name, output_dir)
        
        # Create storyboard after all scenes are generated
        print("\n" + "="*50)
        create_bus_journey_storyboard(output_dir, include_references=True)
        
        print("\nGeneration complete!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if hasattr(pipeline, 'pipeline') and pipeline.pipeline:
            del pipeline.pipeline
        if hasattr(pipeline, 'compel') and pipeline.compel:
            del pipeline.compel
        if hasattr(pipeline, 'controlnet') and pipeline.controlnet:
            del pipeline.controlnet
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()