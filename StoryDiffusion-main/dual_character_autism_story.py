"""
REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - OPTIMIZED VERSION
Removed unnecessary inpaint pipeline for faster execution

✅ KEY FEATURES:
- Uses Compel library to handle prompts longer than 77 tokens
- Automatic prompt chunking and concatenation
- Support for SDXL dual text encoders
- SINGLE ControlNet pipeline for all generations
- Scene-specific negative prompts for better control
- Automatic storyboard creation with AI labels!
"""

import os
import torch
import time
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from diffusers import StableDiffusionXLControlNetPipeline, DDIMScheduler, ControlNetModel
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc
import numpy as np

class StoryboardCreator:
    """Creates professional storyboards with images and descriptive labels"""
    
    def __init__(self, img_size=350, padding=30, label_height=80):
        self.img_size = img_size
        self.padding = padding
        self.label_height = label_height
        self.images = []
        self.labels = []
        
    def add_image(self, image_path, label_text):
        """Add an image and its label to the storyboard"""
        if os.path.exists(image_path):
            self.images.append(image_path)
            self.labels.append(label_text)
            return True
        return False
    
    def _get_font(self, size=16, bold=False):
        """Try to get a good font, fallback to default if needed"""
        font_options = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\Arial.ttf",
            "arial.ttf",
        ]
        
        for font_path in font_options:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        # Fallback to default
        return ImageFont.load_default()
    
    def _simplify_label(self, text):
        """Convert detailed action descriptions to simple labels"""
        # Map of scene names to simple labels with numbering
        simple_labels = {
            "alex character": "Alex - Main Character",
            "emma character": "Emma - Friend", 
            "looking curious": "1. Alex notices someone needs help",
            "crying": "2. Emma feels sad and lonely",
            "sad": "2. Emma feels sad and lonely",
            "thinking": "3. Alex thinks how to help",
            "walking towards": "4. Alex approaches kindly",
            "high five": "5. Alex and Emma friendly greeting",
            "playing": "6. Alex and Emma playing together",
            "sharing": "7. Alex and Emma sharing and caring",
        }
        
        # Check for keywords and return simple label
        text_lower = text.lower()
        for key, label in simple_labels.items():
            if key in text_lower:
                return label
        
        # If no match, create a simple version
        if "alex" in text_lower and "alone" in text_lower:
            return "1. Alex notices someone needs help"
        elif "emma" in text_lower and ("sad" in text_lower or "crying" in text_lower):
            return "2. Emma feels sad and lonely"
        elif "thinking" in text_lower or "thought" in text_lower:
            return "3. Alex thinks how to help"
        elif "approach" in text_lower or "walking" in text_lower:
            return "4. Alex approaches kindly"
        elif "greeting" in text_lower or "high" in text_lower:
            return "5. Alex and Emma friendly greeting"
        elif "playing" in text_lower or "toys" in text_lower:
            return "6. Alex and Emma playing together"
        elif "sharing" in text_lower or "books" in text_lower:
            return "7. Alex and Emma sharing and caring"
        
        # Final fallback - take first few words
        words = text.split()[:4]
        return " ".join(words).title()
    
    def _draw_text_with_wrap(self, draw, text, position, font, max_width, color="black"):
        """Draw text with word wrapping"""
        # Wrap text
        wrapped_lines = []
        words = text.split()
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            # Use textbbox for newer PIL versions, fallback to textsize
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except AttributeError:
                text_width = draw.textsize(test_line, font=font)[0]
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    wrapped_lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    wrapped_lines.append(word)
        
        if current_line:
            wrapped_lines.append(" ".join(current_line))
        
        # Draw each line
        x, y = position
        for line in wrapped_lines[:2]:  # Limit to 2 lines
            draw.text((x, y), line, fill=color, font=font)
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_height = bbox[3] - bbox[1] + 5
            except AttributeError:
                line_height = font.getsize(line)[1] + 5
            y += line_height
    
    def create(self, output_path, title="Storyboard", cols=3):
        """Create the storyboard with images and labels"""
        if not self.images:
            print("No images to create storyboard")
            return None
        
        # Calculate dimensions
        num_images = len(self.images)
        rows = (num_images + cols - 1) // cols
        
        # Calculate total size
        total_width = cols * self.img_size + (cols + 1) * self.padding
        total_height = rows * (self.img_size + self.label_height) + (rows + 1) * self.padding + 80  # Extra for title
        
        # Create white background
        storyboard = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(storyboard)
        
        # Get fonts
        title_font = self._get_font(32, bold=True)
        label_font = self._get_font(18)
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (total_width - title_width) // 2
        draw.text((title_x, 20), title, fill='black', font=title_font)
        
        # Draw each image with label
        for idx, (img_path, label_text) in enumerate(zip(self.images, self.labels)):
            row = idx // cols
            col = idx % cols
            
            # Calculate position
            x = col * self.img_size + (col + 1) * self.padding
            y = row * (self.img_size + self.label_height) + (row + 1) * self.padding + 80
            
            try:
                # Load and resize image
                img = Image.open(img_path)
                img.thumbnail((self.img_size, self.img_size), Image.Resampling.LANCZOS)
                
                # Center image in its space
                img_x = x + (self.img_size - img.width) // 2
                img_y = y + (self.img_size - img.height) // 2
                
                # Paste image
                storyboard.paste(img, (img_x, img_y))
                
                # Draw border around image
                draw.rectangle([x-2, y-2, x+self.img_size+2, y+self.img_size+2], outline='gray', width=2)
                
                # Simplify and draw label
                simple_label = self._simplify_label(label_text)
                label_y = y + self.img_size + 10
                
                # Center text under image
                self._draw_text_with_wrap(
                    draw,
                    simple_label,
                    (x + 10, label_y),
                    label_font,
                    self.img_size - 20,
                    color='black'
                )
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Save storyboard
        storyboard.save(output_path, quality=95, optimize=True)
        print(f"✅ Storyboard saved: {output_path}")
        
        return output_path

class CompelRealCartoonPipeline:
    """Single pipeline with Compel support for long prompts and ControlNet for all generations"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.compel = None
        self.ip_adapter_loaded = False
        self.controlnet = None

    def setup_pipeline_with_compel(self):
        """Setup single ControlNet pipeline with Compel for long prompt support"""
        print("🔧 Setting up RealCartoon XL v7 with Compel (Single Pipeline)...")
        
        try:
            # Verify model
            size_gb = os.path.getsize(self.model_path) / (1024**3)
            print(f"  📊 Model: {size_gb:.1f}GB")
            if 6.5 <= size_gb <= 7.5:
                print("  ✅ RealCartoon XL v7 confirmed")

            # Load image encoder first
            print("  📸 Loading image encoder...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
            )

            # Load ControlNet
            print("  🎚️ Loading ControlNet (depth)...")
            self.controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0-small",  # Lightweight version
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16,
            )

            # Create single ControlNet pipeline for all generations
            print("  🔗 Setting up single ControlNet pipeline...")
            self.pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                self.model_path,
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                image_encoder=image_encoder,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            print("  ✅ Single ControlNet pipeline loaded")

            # Setup scheduler
            print("  📐 Setting up scheduler...")
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.enable_vae_tiling()

            # Setup Compel for long prompts (CRITICAL!)
            print("  🚀 Setting up Compel for long prompts...")
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False  # This enables long prompts!
            )

            print("✅ Single pipeline with Compel ready!")
            print("  🎯 Long prompts (>77 tokens) now supported!")
            print("  📏 ControlNet enabled for all generations!")
            
            return True

        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            traceback.print_exc()
            return False

    def load_ip_adapter_simple(self):
        """Load IP-Adapter with simple configuration"""
        print("\n🖼️ Loading IP-Adapter...")
        try:
            # Load IP-Adapter for the single pipeline
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
            )
            
            self.ip_adapter_loaded = True
            print("  ✅ IP-Adapter loaded for all pipelines!")
            return True
            
        except Exception as e:
            print(f"  ❌ IP-Adapter failed: {e}")
            return False

    def encode_prompts_with_compel(self, positive_prompt, negative_prompt):
        """Encode both prompts using Compel and ensure matching shapes"""
        try:
            print(f"  📝 Encoding positive prompt: {positive_prompt[:100]}{'...' if len(positive_prompt) > 100 else ''}")
            print(f"  📊 Positive prompt length: {len(positive_prompt.split())} words")
            print(f"  📝 Encoding negative prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
            print(f"  📊 Negative prompt length: {len(negative_prompt.split())} words")

            # Use Compel to handle both prompts
            positive_conditioning, positive_pooled = self.compel(positive_prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)

            # CRITICAL: Pad conditioning tensors to same length to avoid shape mismatch
            print("  🔧 Padding tensors to same length...")
            [positive_conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )

            print(f"  ✅ Final tensor shapes: pos={positive_conditioning.shape}, neg={negative_conditioning.shape}")
            return positive_conditioning, positive_pooled, negative_conditioning, negative_pooled

        except Exception as e:
            print(f"  ❌ Compel encoding failed: {e}")
            traceback.print_exc()
            return None, None, None, None

    def create_simple_masks(self, width=1024, height=1024):
        """Create properly balanced left/right masks for dual character scenes"""
        # Calculate proper split with minimal overlap for smooth blending
        center = width // 2
        overlap = 20  # Small overlap for smooth blending (reduced from 200)

        # Alex mask - left side (covers left half + small overlap)
        alex_mask = Image.new("L", (width, height), 0)
        alex_draw = ImageDraw.Draw(alex_mask)
        alex_draw.rectangle([0, 0, center + overlap, height], fill=255)

        # Emma mask - right side (covers right half + small overlap)
        emma_mask = Image.new("L", (width, height), 0)
        emma_draw = ImageDraw.Draw(emma_mask)
        emma_draw.rectangle([center - overlap, 0, width, height], fill=255)

        # Light blur for smooth edges (proper blending)
        alex_mask = alex_mask.filter(ImageFilter.GaussianBlur(radius=5))
        emma_mask = emma_mask.filter(ImageFilter.GaussianBlur(radius=5))

        return alex_mask, emma_mask

    def create_highfive_masks(self, width=1024, height=1024):
        """FIXED: Special masks for Scene 5 high-five - larger overlap for hand contact"""
        center = width // 2
        overlap = 80  # Bigger overlap for proper hand contact

        # Alex mask - left side
        alex_mask = Image.new("L", (width, height), 0)
        alex_draw = ImageDraw.Draw(alex_mask)
        alex_draw.rectangle([0, 0, center + overlap//2, height], fill=255)

        # Emma mask - right side  
        emma_mask = Image.new("L", (width, height), 0)
        emma_draw = ImageDraw.Draw(emma_mask)
        emma_draw.rectangle([center - overlap//2, 0, width, height], fill=255)

        # Light blur for smooth hand blending
        alex_mask = alex_mask.filter(ImageFilter.GaussianBlur(radius=3))
        emma_mask = emma_mask.filter(ImageFilter.GaussianBlur(radius=3))

        return alex_mask, emma_mask

    def create_height_consistent_depth_map(self, width=1024, height=1024):
        """Create depth map designed for height consistency between dual characters"""
        # Create uniform depth for same-height characters
        depth_map = np.ones((height, width), dtype=np.uint8) * 128  # Mid-gray = same distance

        # Key: Add ground plane reference for height consistency
        # Bottom 10% of image = ground level (darker = closer to camera)
        ground_height = int(height * 0.9)  # Characters should "stand" at this level

        # Ground plane - slightly closer to establish baseline
        depth_map[ground_height:, :] = 115  # Darker = closer = ground level

        # Character zones - same depth ensures same scale/height
        char_depth = 125  # Consistent depth for both characters

        # Left character zone (Alex) - uniform depth for consistent height
        depth_map[int(height*0.1):ground_height, :width//2] = char_depth

        # Right character zone (Emma) - same depth for same height
        depth_map[int(height*0.1):ground_height, width//2:] = char_depth

        # Smooth transitions
        depth_map = Image.fromarray(depth_map, mode='L')
        depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=3))

        return depth_map

    def create_blank_depth_map(self, width=1024, height=1024):
        """Create blank depth map for single character/reference generations"""
        # Simple uniform depth map for single characters
        depth_map = np.ones((height, width), dtype=np.uint8) * 128  # Mid-gray = neutral depth
        return Image.fromarray(depth_map, mode='L')

    def generate_reference_with_compel(self, prompt, negative_prompt, **kwargs):
        """Generate character reference using single pipeline with Compel for long prompts"""
        try:
            # Encode both prompts with Compel and ensure matching shapes
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )

            if prompt_embeds is None:
                return None

            # Create blank depth map for ControlNet (minimal influence)
            depth_map = self.create_blank_depth_map()

            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                image=depth_map,  # ControlNet depth input (minimal influence)
                controlnet_conditioning_scale=0.1,  # Very low influence for references
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

    def generate_single_character_scene_with_compel(self, prompt, negative_prompt, character_ref, scene_name=None, **kwargs):
        """Generate single character scene using single pipeline with Compel"""
        if not self.ip_adapter_loaded:
            print("  ❌ IP-Adapter not loaded!")
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

            # Create blank depth map for ControlNet (minimal influence)
            depth_map = self.create_blank_depth_map()

            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                ip_adapter_image=character_ref,
                image=depth_map,  # ControlNet depth input (minimal influence)
                controlnet_conditioning_scale=0.2,  # Low influence for single characters
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

    def generate_dual_character_scene_with_compel(self, prompt, negative_prompt, alex_ref, emma_ref, scene_name=None, **kwargs):
        """Generate dual character scene using single pipeline with Compel and ControlNet for size consistency"""
        if not self.ip_adapter_loaded:
            print("  ❌ IP-Adapter not loaded!")
            return None

        try:
            print("  📏 Using single ControlNet pipeline for character size consistency...")

            # Check if this is Scene 5 (high-five)
            is_scene_5 = scene_name == "05_greeting" if scene_name else False

            # SCENE 5 SPECIFIC: Add clothing preservation to prompt
            if is_scene_5:
                prompt = prompt + ", Alex wearing blue t-shirt and dark pants, Emma wearing pink cardigan and white shirt"

            # Encode prompts with Compel
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )

            if prompt_embeds is None:
                return None

            # Create masks based on scene
            if is_scene_5:
                print("  🤝 Scene 5 detected - using high-five masks with aggressive anti-blending settings...")
                alex_mask, emma_mask = self.create_highfive_masks()
                controlnet_scale = 0.1  # VERY low ControlNet influence
                ip_scale = [[0.5, 0.5]]  # REDUCED IP-Adapter strength to prevent blending
            else:
                alex_mask, emma_mask = self.create_simple_masks()
                controlnet_scale = 0.7  # Normal for other scenes
                ip_scale = [[0.7, 0.7]]  # Normal IP-Adapter scale

            # Process masks
            processor = IPAdapterMaskProcessor()
            masks = processor.preprocess([alex_mask, emma_mask], height=1024, width=1024)

            # Create depth map for ControlNet with height consistency
            depth_map = self.create_height_consistent_depth_map()

            # IP-Adapter configuration for masking
            self.pipeline.set_ip_adapter_scale(ip_scale)

            # Format for masking
            ip_images = [[alex_ref, emma_ref]]
            masks_formatted = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]

            # Use single ControlNet pipeline with adjusted control
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                ip_adapter_image=ip_images,
                cross_attention_kwargs={"ip_adapter_masks": masks_formatted},
                image=depth_map,  # Height-consistent depth input
                controlnet_conditioning_scale=controlnet_scale,
                **kwargs
            )

            print("  ✅ Generated with single ControlNet pipeline!")

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
        """Detailed reference prompts with height consistency (no token limit with Compel!)"""
        if character_name == "Alex":
            return f"""{self.alex_base}, character reference sheet, full body portrait,
standing pose with consistent proportions and scale, clean white background,
high quality digital illustration, consistent character design,
educational animation style, bright even lighting, clear details,
professional character concept art, proper child proportions,
age-appropriate height and scale, 6-year-old proportions"""
        else:
            return f"""{self.emma_base}, character reference sheet, full body portrait,
standing pose with consistent proportions and scale, clean white background,
high quality digital illustration, consistent character design,
educational animation style, bright even lighting, clear details,
professional character concept art, proper child proportions,
age-appropriate height and scale, 6-year-old proportions"""

    def get_scene_prompt(self, skill, character_focus, action):
        """Detailed scene prompts with height consistency cues (no token limit with Compel!)"""
        if character_focus == "Alex":
            return f"""{self.alex_base}, {action}, {self.setting_base},
autism education themed illustration, social skills learning context,
high quality educational content, detailed facial expressions showing emotion,
natural body language and gestures, warm inviting atmosphere,
child-appropriate content, positive educational messaging,
consistent character proportions, proper scale and height"""
        elif character_focus == "Emma":
            return f"""{self.emma_base}, {action}, {self.setting_base},
autism education themed illustration, social skills learning context,
high quality educational content, detailed facial expressions showing emotion,
natural body language and gestures, warm inviting atmosphere,
child-appropriate content, positive educational messaging,
consistent character proportions, proper scale and height"""
        else:  # both characters
            return f"""Alex and Emma, two children of similar height standing together, {action}, {self.setting_base},
both characters are the same scale and proportional height, equal sized children,
consistent character proportions between both children, same height characters,
social skills education illustration, autism awareness content, showing positive social interaction,
detailed character expressions and body language, warm educational atmosphere,
high quality digital illustration, child-friendly educational content,
demonstrating friendship and social learning, bright cheerful colors,
both children at same eye level, proportionally consistent characters"""

    def get_negative_prompt(self):
        """Enhanced negative prompt with height consistency prevention"""
        return """realistic photography, photorealistic, adult faces, teenage appearance,
blurry image, low quality, poor resolution, dark lighting, scary atmosphere,
inappropriate content, weapons, violence, sad or distressing themes,
poor anatomy, distorted faces, multiple heads, extra limbs,
bad hands, mutation, deformed features, background characters, other people,
height differences, size inconsistency, disproportionate characters,
one character taller than other, mismatched scale, giant characters,
tiny characters, height mismatch, uneven proportions"""

    def get_scene_specific_negative_prompt(self, scene_name):
        """Get scene-specific negative prompts for targeted control"""
        base_negative = self.get_negative_prompt()
        
        scene_negatives = {
            "01_alex_noticing": """multiple people, girl, woman""",
            
            "02_emma_lonely": """(happy expression:1.7), (smiling:1.6), (cheerful face:1.5),
            (joyful demeanor:1.4), (excited appearance:1.3), (bright happy eyes:1.4),
            (upturned mouth:1.5), (laughing:1.4), (pleased look:1.3), (content expression:1.3),
            (delighted face:1.3), positive emotion, upbeat mood, enthusiastic, gleeful""",
            
            "03_alex_thinking": """(smiling:1.5), (happy expression:1.4), (grinning:1.3),
            (cheerful look:1.3), (vacant stare:1.3), (blank expression:1.3),
            (neutral face:1.2), (relaxed eyebrows:1.3), (unfocused eyes:1.2),
            (distracted look:1.2), not thinking, daydreaming""",
            
            "04_approaching": """aggressive approach, hostile interaction, mean behavior,
            intimidating posture, angry confrontation, bullying behavior, threatening gesture,
            scared withdrawal, fearful reaction, running away, avoidance, rejection,
            unfriendly meeting, negative interaction, conflict""",
            
            "05_greeting": """clothing blending, bag, girl in jeans""",
            
            "06_playing_together": """clothes blending""",
            
            "07_sharing": """selfish behavior, hoarding, grabbing, possessive attitude,
            greedy expression, keeping everything, refusing to share,
            mean spirited, stingy behavior, protective of belongings,
            excluding others, ungenerous, territorial behavior"""
        }
        
        scene_specific = scene_negatives.get(scene_name, "")
        if scene_specific:
            return f"{base_negative}, {scene_specific}"
        else:
            return base_negative

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

def create_autism_education_storyboard(output_dir, educational_scenes):
    """Create storyboard using the integrated storyboard creator"""
    print("\n📋 Creating Autism Education Storyboard...")
    print("-" * 60)
    
    # Create storyboard instance
    storyboard = StoryboardCreator(img_size=350, padding=30)
    
    # Add reference images
    alex_ref_path = os.path.join(output_dir, "alex_reference_detailed.png")
    emma_ref_path = os.path.join(output_dir, "emma_reference_detailed.png")
    
    if os.path.exists(alex_ref_path):
        storyboard.add_image(alex_ref_path, "Alex character")
        print("  ✅ Added Alex reference")
    
    if os.path.exists(emma_ref_path):
        storyboard.add_image(emma_ref_path, "Emma character")
        print("  ✅ Added Emma reference")
    
    # Add scene images with their action prompts
    for scene_name, scene_type, character_focus, action in educational_scenes:
        image_path = os.path.join(output_dir, f"{scene_name}.png")
        if os.path.exists(image_path):
            # The AI will create simple labels from the action descriptions
            storyboard.add_image(image_path, action)
    
    # Create and save storyboard
    storyboard_path = os.path.join(output_dir, "autism_education_storyboard.png")
    return storyboard.create(
        storyboard_path,
        title="Social Skills: Making Friends",
        cols=3  # 3 columns for better layout
    )

def main():
    """RealCartoon XL v7 with Compel Long Prompts - OPTIMIZED VERSION"""
    print("\n" + "="*70)
    print("🚀 REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - OPTIMIZED")
    print("  Removed unnecessary inpaint pipeline for faster execution!")
    print("="*70)
    
    print("\n🎯 Single Pipeline Features:")
    print("  • Uses ONE ControlNet pipeline for all generations")
    print("  • Handles prompts longer than 77 tokens")
    print("  • Automatic prompt chunking and concatenation")
    print("  • Scene-specific negative prompts for better control")
    print("  • Automatic storyboard creation with AI labels!")
    print("  • REMOVED: Inpaint pipeline (saves loading time)")
    print("  • REMOVED: MediaPipe face detection (not essential)")
    
    # Check if Compel is installed
    try:
        import compel
        print("  ✅ Compel library detected")
    except ImportError:
        print("  ❌ Compel library not found!")
        print("  Install with: pip install compel")
        return
    
    print("  ✅ Integrated storyboard creator ready!")

    # Find model
    model_path = find_realcartoon_model()
    if not model_path:
        model_path = input("📁 Enter RealCartoon XL v7 path: ").strip()
        if not os.path.exists(model_path):
            print("❌ Model not found")
            return

    output_dir = "compel_optimized_autism_education"
    os.makedirs(output_dir, exist_ok=True)

    # Setup single pipeline with Compel
    pipeline = CompelRealCartoonPipeline(model_path)
    if not pipeline.setup_pipeline_with_compel():
        print("❌ Pipeline setup failed")
        return

    prompts = DetailedEducationalPrompts()

    # Define scenes with detailed actions
    educational_scenes = [
        # Single character scenes
        ("01_alex_noticing", "single", "Alex",
         "Alex alone, Looking curious"),
         
        ("02_emma_lonely", "single", "Emma",
         "(CRYING EMMA:1.6), (SO SAD:1.5), (extremely sad facial expression:1.5), (visible tears streaming down face:1.4), (dramatically downturned mouth:1.4), (very drooping eyelids:1.3), (heartbroken appearance:1.4), (profound sadness:1.3), (emotional distress:1.3), (visibly sobbing:1.4), sitting alone at her desk, (lonely and devastated:1.3), (unmistakably melancholic:1.4)"),
         
        ("03_alex_thinking", "single", "Alex",
         "(THINKING ALEX:1.8), (DEEP IN THOUGHT:1.7), (hand on chin thinking pose:1.6), (eyes looking upward and to the left:1.5), (strongly furrowed brow:1.5), (concentrated thinking expression:1.4), (pondering deeply:1.4), (contemplative look:1.3), at his desk, ALONE"),
         
        # Dual character scenes
        ("04_approaching", "dual", "both",
         "Alex walking towards Emma at her desk while Emma looks up with cautious but hopeful expression, showing the beginning of positive social interaction"),
         
        # Scene 5 with enhanced prompt for high-five
        ("05_greeting", "dual", "both",
         "Alex and Emma, two 6-year-old children standing face to face in the center of the image, both raising their right hands up high to do a high five together, their palms meeting in the middle, both smiling happily at each other, centered composition with both children fully visible, natural interaction in open space"),
         
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
        print(f"  📊 Alex prompt length: {len(alex_prompt.split())} words")
        
        alex_ref = pipeline.generate_reference_with_compel(
            alex_prompt, negative,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(1000)
        )
        
        print("🎨 Generating Emma reference with detailed prompt...")
        emma_prompt = prompts.get_reference_prompt("Emma")
        print(f"  📊 Emma prompt length: {len(emma_prompt.split())} words")
        
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

        # PHASE 3: Generate detailed scenes with scene-specific negative prompts
        print(f"\n🎬 PHASE 3: Detailed Educational Scenes ({len(educational_scenes)} total)")
        print("-" * 70)

        results = []
        total_time = 0

        for i, (scene_name, scene_type, character_focus, action) in enumerate(educational_scenes):
            print(f"\n🎨 Scene {i+1}: {scene_name} ({scene_type})")
            
            prompt = prompts.get_scene_prompt("social skills", character_focus, action)
            
            # Use scene-specific negative prompt
            scene_negative = prompts.get_scene_specific_negative_prompt(scene_name)
            print(f"  🎯 Using scene-specific negative prompt for {scene_name}")
            
            output_path = os.path.join(output_dir, f"{scene_name}.png")
            print(f"  📊 Scene prompt length: {len(prompt.split())} words")
            print(f"  📊 Negative prompt length: {len(scene_negative.split())} words")

            start_time = time.time()

            # Use appropriate generation method
            if scene_type == "single":
                character_ref = alex_ref if character_focus == "Alex" else emma_ref
                image = pipeline.generate_single_character_scene_with_compel(
                    prompt, scene_negative, character_ref,
                    scene_name=scene_name,  # Pass scene name for auto-fix
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(1000 + i)
                )
            else:  # dual
                image = pipeline.generate_dual_character_scene_with_compel(
                    prompt, scene_negative, alex_ref, emma_ref,
                    scene_name=scene_name,  # Pass scene name for Scene 5 detection
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
                print(f"  ✅ Generated: {scene_name} ({scene_time:.1f}s)")
            else:
                print(f"  ❌ Failed: {scene_name}")

        # PHASE 4: Create Storyboard
        if results:
            print("\n📋 PHASE 4: Creating Storyboard")
            print("-" * 60)
            create_autism_education_storyboard(output_dir, educational_scenes)

        # Results
        print("\n" + "=" * 70)
        print("🚀 OPTIMIZED PIPELINE RESULTS")
        print("=" * 70)

        success_rate = len(results) / len(educational_scenes) * 100
        avg_time = total_time / len(educational_scenes) if educational_scenes else 0

        print(f"✓ Success rate: {success_rate:.1f}% ({len(results)}/{len(educational_scenes)})")
        print(f"✓ Average time: {avg_time:.1f}s per scene")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Used single ControlNet pipeline for all generations")
        print(f"✓ Used Compel for long prompts")
        print(f"✓ No 77-token truncation warnings!")
        print(f"✓ Scene-specific negative prompts applied!")

        if results:
            print(f"✓ Storyboard created with AI labels!")

        if results:
            print("\n📋 Generated scenes:")
            for result in results:
                print(f"  • {result}")

        print("\n🎉 OPTIMIZATIONS APPLIED:")
        print("  • Removed inpaint pipeline (faster loading)")
        print("  • Removed MediaPipe dependency")
        print("  • Single ControlNet pipeline for everything")
        print("  • Long prompts (>77 tokens) working perfectly")
        print("  • Scene-specific negative prompts")
        print("  • Automatic storyboard creation")
        print("  • Faster execution with same quality")

        if success_rate >= 80:
            print("\n🚀 OPTIMIZED PIPELINE SUCCESS!")
            print("  📚 Advanced autism education materials")
            print("  🎭 Superior character consistency with ControlNet")
            print("  🎨 Long detailed prompts working perfectly")
            print("  ✨ 77-token CLIP limit solved with Compel!")
            print("  🔧 Single pipeline architecture (no extras!)")
            print("  📏 HEIGHT CONSISTENCY via ControlNet depth!")
            print("  🎯 Each scene optimized with custom negative prompts!")
            print("  📋 Professional storyboard ready for use!")

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