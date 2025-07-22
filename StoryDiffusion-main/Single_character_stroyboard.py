"""
REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - OPTIMIZED VERSION
ENHANCED WITH AUTISM-FRIENDLY STORYBOARD

Enhanced storyboard creation to match Alex morning routine style:
- Clear step-by-step visual sequence with card design
- Descriptive labels beneath each image
- Professional layout with shadows and proper spacing
- Text wrapping for longer descriptions
- Consistent styling with morning routine version
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

class AutismFriendlyStoryboardCreator:
    """Creates autism-friendly storyboards matching Alex morning routine style exactly"""
    
    def __init__(self):
        # Configuration exactly matching morning routine style
        self.img_size = 320
        self.padding_horizontal = 40  # Reduced padding
        self.padding_vertical = 30    # Reduced padding
        self.label_spacing = 10       # Space between image and label
        self.title_height = 80
        self.footer_height = 60
        self.card_padding = 15
        
        # Define autism-friendly labels for social skills story - exactly matching morning routine format
        self.autism_friendly_labels = {
            "alex_reference_detailed": "Alex - Main Character",
            "emma_reference_detailed": "Emma - Friend", 
            "01_alex_noticing": "Step 1: Alex notices Emma looks sad",
            "02_emma_lonely": "Step 2: Emma feels lonely and upset",
            "03_alex_thinking": "Step 3: Alex thinks about how to help",
            "04_approaching": "Step 4: Alex walks over to Emma kindly",
            "05_greeting": "Step 5: Alex and Emma say hello with a high-five",
            "06_playing_together": "Step 6: Alex and Emma play together happily",
            "07_sharing": "Step 7: Alex and Emma share books and learn together"
        }
    
    def _get_font(self, size=16, bold=False):
        """Try to get good fonts with fallbacks - exactly matching morning routine"""
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
        
        return ImageFont.load_default()
    
    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within specified width"""
        words = text.split()
        lines = []
        current_line = []
        
        # Create temporary image for text measurement
        temp_img = Image.new('RGB', (100, 100))
        temp_draw = ImageDraw.Draw(temp_img)
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = temp_draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except AttributeError:
                text_width = temp_draw.textsize(test_line, font=font)[0]
            
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
    
    def create_social_skills_storyboard(self, output_dir, educational_scenes):
        """Create autism-friendly storyboard exactly matching morning routine style"""
        print("\n📋 Creating Autism-Friendly Social Skills Storyboard...")
        print("-" * 60)
        
        # Collect scene images (skip reference images for main storyboard) - matching morning routine approach
        images = []
        labels = []
        
        # Skip reference images and only include story scenes - exactly like morning routine
        story_scenes = [scene for scene in educational_scenes if not scene[0].endswith('_reference_detailed')]
        
        for scene_name, _, _, action in story_scenes:
            scene_path = os.path.join(output_dir, f"{scene_name}.png")
            if os.path.exists(scene_path):
                img = Image.open(scene_path)
                images.append(img)
                
                if scene_name in self.autism_friendly_labels:
                    labels.append(self.autism_friendly_labels[scene_name])
                else:
                    # Fallback label
                    labels.append(f"Step: {scene_name.replace('_', ' ').title()}")
                print(f"  ✅ Added: {self.autism_friendly_labels.get(scene_name, scene_name)}")
        
        if not images:
            print("  ❌ No images found for storyboard")
            return None
        
        # Get fonts - exactly matching morning routine
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
        
        # Calculate grid layout - exactly matching morning routine
        num_images = len(images)
        cols = 3  # Match morning routine style exactly
        rows = (num_images + cols - 1) // cols
        
        # Calculate maximum label height needed for wrapped text - exactly matching morning routine
        text_area_width = self.img_size  # Text area same width as image
        max_label_height = 0
        
        for label in labels:
            wrapped_lines = self._wrap_text(label, label_font, text_area_width)
            line_height = 22  # Height per line of text - exactly matching morning routine
            total_text_height = len(wrapped_lines) * line_height
            max_label_height = max(max_label_height, total_text_height)
        
        # Calculate canvas size - exactly matching morning routine
        card_width = self.img_size + 2 * self.card_padding
        card_height = self.img_size + max_label_height + self.label_spacing + 2 * self.card_padding
        
        canvas_width = cols * card_width + (cols + 1) * self.padding_horizontal
        canvas_height = self.title_height + rows * card_height + (rows + 1) * self.padding_vertical + self.footer_height
        
        # Create canvas with light background - exactly matching morning routine
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='#F5F5F5')
        draw = ImageDraw.Draw(canvas)
        
        # Add title with background - exactly matching morning routine
        title = "Alex and Emma's Friendship Story"  # Updated title to match social skills theme
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (canvas_width - title_width) // 2
        title_y = 25
        
        # Draw title background - exactly matching morning routine blue
        draw.rectangle([0, 0, canvas_width, self.title_height], fill='#2196F3')
        draw.text((title_x, title_y), title, fill='white', font=title_font)
        
        # Place images in grid with card design - exactly matching morning routine
        for i, (img, label) in enumerate(zip(images, labels)):
            row = i // cols
            col = i % cols
            
            # Calculate position - exactly matching morning routine
            x = self.padding_horizontal + col * (card_width + self.padding_horizontal)
            y = self.title_height + self.padding_vertical + row * (card_height + self.padding_vertical)
            
            # Draw shadow effect - exactly matching morning routine
            shadow_offset = 3
            draw.rectangle([x + shadow_offset, y + shadow_offset, 
                           x + card_width + shadow_offset, y + card_height + shadow_offset], 
                          fill='#E0E0E0', outline=None)
            
            # Draw card - exactly matching morning routine
            draw.rectangle([x, y, x + card_width, y + card_height], 
                          fill='white', outline='#E0E0E0', width=2)
            
            # Resize and paste image - exactly matching morning routine
            img_resized = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            img_x = x + self.card_padding
            img_y = y + self.card_padding
            canvas.paste(img_resized, (img_x, img_y))
            
            # Draw wrapped text - exactly matching morning routine
            text_start_y = y + self.card_padding + self.img_size + self.label_spacing
            wrapped_lines = self._wrap_text(label, label_font, text_area_width)
            
            for line_idx, line in enumerate(wrapped_lines):
                # Center each line - exactly matching morning routine
                line_bbox = draw.textbbox((0, 0), line, font=label_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = x + self.card_padding + (self.img_size - line_width) // 2
                line_y = text_start_y + line_idx * 22
                
                draw.text((line_x, line_y), line, fill='#333333', font=label_font)
        
        # Add footer with background - exactly matching morning routine
        footer_y_start = canvas_height - self.footer_height
        draw.rectangle([0, footer_y_start, canvas_width, canvas_height], fill='#E3F2FD')
        
        footer_text = "This storyboard shows how Alex and Emma become friends step by step"
        footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        footer_x = (canvas_width - footer_width) // 2
        footer_y = footer_y_start + (self.footer_height - (footer_bbox[3] - footer_bbox[1])) // 2
        
        draw.text((footer_x, footer_y), footer_text, fill='#555555', font=footer_font)
        
        # Save storyboard - exactly matching morning routine naming convention
        storyboard_path = os.path.join(output_dir, "alex_emma_friendship_autism_friendly_storyboard.png")
        canvas.save(storyboard_path, quality=95, optimize=True)
        
        print(f"\n✅ Autism-Friendly Social Skills Storyboard saved: {storyboard_path}")
        print(f"  📐 Size: {canvas_width}x{canvas_height}px")
        print(f"  📚 Contains {len(images)} images with descriptive labels")
        print(f"  🎯 Designed for autism support with clear visual steps")
        print(f"  🎨 Features: Card design, shadows, numbered steps, text wrapping")
        print(f"  🎨 Style: Exactly matching Alex morning routine storyboard")
        
        return storyboard_path

class StoryboardCreator:
    """Legacy storyboard creator for backwards compatibility"""
    
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
        
        return ImageFont.load_default()
    
    def _simplify_label(self, text):
        """Convert detailed action descriptions to simple labels"""
        simple_labels = {
            "alex character": "Alex - Main Character",
            "emma character": "Emma - Friend",
            "looking curious": "Noticing Someone Needs Help",
            "crying": "Feeling Sad and Lonely",
            "sad": "Feeling Sad and Lonely",
            "thinking": "Thinking How to Help",
            "walking towards": "Approaching Kindly",
            "high five": "Friendly Greeting",
            "playing": "Playing Together",
            "sharing": "Sharing and Caring",
        }
        
        text_lower = text.lower()
        for key, label in simple_labels.items():
            if key in text_lower:
                return label
        
        if "alex" in text_lower and "alone" in text_lower:
            return "Noticing Someone Needs Help"
        elif "emma" in text_lower and ("sad" in text_lower or "crying" in text_lower):
            return "Feeling Sad and Lonely"
        elif "thinking" in text_lower or "thought" in text_lower:
            return "Thinking How to Help"
        elif "approach" in text_lower or "walking" in text_lower:
            return "Approaching Kindly"
        elif "greeting" in text_lower or "high" in text_lower:
            return "Friendly Greeting"
        elif "playing" in text_lower or "toys" in text_lower:
            return "Playing Together"
        elif "sharing" in text_lower or "books" in text_lower:
            return "Sharing and Caring"
        
        words = text.split()[:4]
        return " ".join(words).title()
    
    def _draw_text_with_wrap(self, draw, text, position, font, max_width, color="black"):
        """Draw text with word wrapping"""
        wrapped_lines = []
        words = text.split()
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
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
        """Create the legacy storyboard"""
        if not self.images:
            print("No images to create storyboard")
            return None
        
        num_images = len(self.images)
        rows = (num_images + cols - 1) // cols
        
        total_width = cols * self.img_size + (cols + 1) * self.padding
        total_height = rows * (self.img_size + self.label_height) + (rows + 1) * self.padding + 80
        
        storyboard = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(storyboard)
        
        title_font = self._get_font(32, bold=True)
        label_font = self._get_font(18)
        
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (total_width - title_width) // 2
        draw.text((title_x, 20), title, fill='black', font=title_font)
        
        for idx, (img_path, label_text) in enumerate(zip(self.images, self.labels)):
            row = idx // cols
            col = idx % cols
            
            x = col * self.img_size + (col + 1) * self.padding
            y = row * (self.img_size + self.label_height) + (row + 1) * self.padding + 80
            
            try:
                img = Image.open(img_path)
                img.thumbnail((self.img_size, self.img_size), Image.Resampling.LANCZOS)
                
                img_x = x + (self.img_size - img.width) // 2
                img_y = y + (self.img_size - img.height) // 2
                
                storyboard.paste(img, (img_x, img_y))
                
                draw.rectangle([x-2, y-2, x+self.img_size+2, y+self.img_size+2], outline='gray', width=2)
                
                simple_label = self._simplify_label(label_text)
                label_y = y + self.img_size + 10
                
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
                "diffusers/controlnet-depth-sdxl-1.0-small",
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
        center = width // 2
        overlap = 20
        
        # Alex mask - left side
        alex_mask = Image.new("L", (width, height), 0)
        alex_draw = ImageDraw.Draw(alex_mask)
        alex_draw.rectangle([0, 0, center + overlap, height], fill=255)
        
        # Emma mask - right side
        emma_mask = Image.new("L", (width, height), 0)
        emma_draw = ImageDraw.Draw(emma_mask)
        emma_draw.rectangle([center - overlap, 0, width, height], fill=255)
        
        # Light blur for smooth edges
        alex_mask = alex_mask.filter(ImageFilter.GaussianBlur(radius=5))
        emma_mask = emma_mask.filter(ImageFilter.GaussianBlur(radius=5))
        
        return alex_mask, emma_mask

    def create_highfive_masks(self, width=1024, height=1024):
        """Special masks for Scene 5 high-five - larger overlap for hand contact"""
        center = width // 2
        overlap = 80
        
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
        depth_map = np.ones((height, width), dtype=np.uint8) * 128
        
        # Add ground plane reference for height consistency
        ground_height = int(height * 0.9)
        depth_map[ground_height:, :] = 115
        
        char_depth = 125
        depth_map[int(height*0.1):ground_height, :width//2] = char_depth
        depth_map[int(height*0.1):ground_height, width//2:] = char_depth
        
        # Smooth transitions
        depth_map = Image.fromarray(depth_map, mode='L')
        depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=3))
        
        return depth_map

    def create_blank_depth_map(self, width=1024, height=1024):
        """Create blank depth map for single character/reference generations"""
        depth_map = np.ones((height, width), dtype=np.uint8) * 128
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
                image=depth_map,
                controlnet_conditioning_scale=0.1,
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
                image=depth_map,
                controlnet_conditioning_scale=0.2,
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
                controlnet_scale = 0.1
                ip_scale = [[0.5, 0.5]]
            else:
                alex_mask, emma_mask = self.create_simple_masks()
                controlnet_scale = 0.7
                ip_scale = [[0.7, 0.7]]
            
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
                image=depth_map,
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
    """Create both autism-friendly and legacy storyboards"""
    print("\n📋 Creating Both Storyboard Versions...")
    print("-" * 60)
    
    # Create autism-friendly storyboard (new style)
    autism_creator = AutismFriendlyStoryboardCreator()
    autism_storyboard = autism_creator.create_social_skills_storyboard(output_dir, educational_scenes)
    
    # Also create legacy storyboard for backwards compatibility
    legacy_creator = StoryboardCreator(img_size=350, padding=30)
    
    # Add reference images
    alex_ref_path = os.path.join(output_dir, "alex_reference_detailed.png")
    emma_ref_path = os.path.join(output_dir, "emma_reference_detailed.png")
    
    if os.path.exists(alex_ref_path):
        legacy_creator.add_image(alex_ref_path, "Alex character")
        print("  ✅ Added Alex reference to legacy storyboard")
    
    if os.path.exists(emma_ref_path):
        legacy_creator.add_image(emma_ref_path, "Emma character")
        print("  ✅ Added Emma reference to legacy storyboard")
    
    # Add scene images with their action prompts
    for scene_name, _, _, action in educational_scenes:
        image_path = os.path.join(output_dir, f"{scene_name}.png")
        if os.path.exists(image_path):
            legacy_creator.add_image(image_path, action)
    
    # Create legacy storyboard
    legacy_storyboard_path = os.path.join(output_dir, "autism_education_storyboard_legacy.png")
    legacy_storyboard = legacy_creator.create(
        legacy_storyboard_path,
        title="Social Skills: Making Friends",
        cols=3
    )
    
    return autism_storyboard, legacy_storyboard

def main():
    """RealCartoon XL v7 with Compel Long Prompts - OPTIMIZED VERSION"""
    print("\n" + "="*70)
    print("🚀 REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - OPTIMIZED")
    print("  Enhanced with Autism-Friendly Storyboard!")
    print("="*70)
    
    print("\n🎯 Enhanced Features:")
    print("  • Uses ONE ControlNet pipeline for all generations")
    print("  • Handles prompts longer than 77 tokens")
    print("  • Automatic prompt chunking and concatenation")
    print("  • Scene-specific negative prompts for better control")
    print("  • NEW: Autism-friendly storyboard with card design!")
    print("  • NEW: Clear descriptive labels and professional layout")
    print("  • LEGACY: Original storyboard also created for compatibility")
    
    # Check if Compel is installed
    try:
        import compel
        print("  ✅ Compel library detected")
    except ImportError:
        print("  ❌ Compel library not found!")
        print("  Install with: pip install compel")
        return
    
    # Find model
    model_path = find_realcartoon_model()
    if not model_path:
        model_path = input("📁 Enter RealCartoon XL v7 path: ").strip()
        if not os.path.exists(model_path):
            print("❌ Model not found")
            return

    output_dir = "compel_optimized_autism_education_enhanced"
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
                    scene_name=scene_name,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(1000 + i)
                )
            else:  # dual
                image = pipeline.generate_dual_character_scene_with_compel(
                    prompt, scene_negative, alex_ref, emma_ref,
                    scene_name=scene_name,
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

        # PHASE 4: Create Enhanced Storyboards
        if results:
            print("\n📋 PHASE 4: Creating Enhanced Storyboards")
            print("-" * 60)
            
            autism_storyboard, legacy_storyboard = create_autism_education_storyboard(output_dir, educational_scenes)
            
            if autism_storyboard:
                print(f"\n✅ Autism-Friendly Storyboard: {autism_storyboard}")
            if legacy_storyboard:
                print(f"✅ Legacy Storyboard: {legacy_storyboard}")

        # Results
        print("\n" + "=" * 70)
        print("🚀 ENHANCED PIPELINE RESULTS")
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
            print(f"✓ Enhanced autism-friendly storyboard created!")
            print(f"✓ Legacy storyboard also created for compatibility!")

        if results:
            print("\n📋 Generated scenes:")
            for result in results:
                print(f"  • {result}")

        print("\n🎉 ENHANCEMENTS APPLIED:")
        print("  • Autism-friendly storyboard exactly matching morning routine style")
        print("  • Clear descriptive labels and professional layout")
        print("  • Text wrapping for longer descriptions")
        print("  • Shadow effects and proper spacing")
        print("  • Both new and legacy storyboard versions")
        print("  • Single ControlNet pipeline for everything")
        print("  • Long prompts (>77 tokens) working perfectly")
        print("  • Scene-specific negative prompts")
        print("  • Faster execution with same quality")
        print("  • Consistent visual style with Alex morning routine")

        if success_rate >= 80:
            print("\n🚀 ENHANCED PIPELINE SUCCESS!")
            print("  📚 Advanced autism education materials")
            print("  🎭 Superior character consistency with ControlNet")
            print("  🎨 Long detailed prompts working perfectly")
            print("  ✨ 77-token CLIP limit solved with Compel!")
            print("  🔧 Single pipeline architecture (no extras!)")
            print("  📏 HEIGHT CONSISTENCY via ControlNet depth!")
            print("  🎯 Each scene optimized with custom negative prompts!")
            print("  📋 TWO professional storyboards ready for use!")
            print("  🎨 Autism-friendly design exactly matching morning routine!")
            print("  🔄 Consistent storyboard style across all projects!")

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