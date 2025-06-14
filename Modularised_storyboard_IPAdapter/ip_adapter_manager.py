"""
IP-Adapter Manager - Handles IP-Adapter integration for character consistency
Ensures same character throughout storyboard sequences
"""

import torch
import numpy as np
from PIL import Image
import cv2


class IPAdapterManager:
    """Manages IP-Adapter for character-consistent generation"""
    
    def __init__(self, base_pipeline, ip_adapter_path=None, device="cuda"):
        print("🎨 Loading IP-Adapter Manager...")
        
        self.base_pipeline = base_pipeline
        self.device = device
        self.ip_adapter_path = ip_adapter_path
        
        # Character reference storage
        self.character_reference_image = None
        self.style_reference_image = None
        self.face_reference_image = None
        
        # Configuration
        self.config = {
            "character_weight": 0.8,
            "style_weight": 0.6,
            "face_weight": 0.9,
            "fallback_to_clip": True
        }
        
        try:
            if ip_adapter_path:
                self._load_ip_adapter()
            else:
                print("⚠️ No IP-Adapter path provided - character consistency will use CLIP only")
                self.available = False
                
        except Exception as e:
            print(f"⚠️ IP-Adapter initialization failed: {e}")
            self.available = False
    
    def _load_ip_adapter(self):
        """Load IP-Adapter models"""
        try:
            # Try different IP-Adapter implementations
            self._try_diffusers_ip_adapter()
            
        except Exception as e:
            try:
                self._try_ip_adapter_plus()
            except Exception as e2:
                print(f"❌ Failed to load any IP-Adapter implementation:")
                print(f"   Diffusers attempt: {e}")
                print(f"   IP-Adapter-Plus attempt: {e2}")
                self.available = False
                return
        
        self.available = True
        print("✅ IP-Adapter loaded successfully")
    
    def _try_diffusers_ip_adapter(self):
        """Try loading IP-Adapter through diffusers"""
        from diffusers import StableDiffusionXLPipeline
        from diffusers.utils import load_image
        
        # Load IP-Adapter into the existing pipeline
        self.base_pipeline.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl.bin"
        )
        
        print("✅ Loaded IP-Adapter via diffusers")
    
    def _try_ip_adapter_plus(self):
        """Try loading IP-Adapter-Plus implementation"""
        # Alternative implementation - you may need to install ip_adapter package
        from ip_adapter import IPAdapterXL
        
        self.ip_adapter = IPAdapterXL(
            self.base_pipeline, 
            image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            ip_ckpt=self.ip_adapter_path,
            device=self.device
        )
        
        print("✅ Loaded IP-Adapter-Plus")
    
    def set_character_reference(self, reference_image, reference_type="character"):
        """Set reference image for character consistency"""
        if isinstance(reference_image, str):
            # Load from path
            reference_image = Image.open(reference_image).convert("RGB")
        elif not isinstance(reference_image, Image.Image):
            # Convert from other formats (numpy, tensor, etc.)
            reference_image = Image.fromarray(reference_image).convert("RGB")
        
        # Preprocess reference image
        processed_image = self._preprocess_reference_image(reference_image)
        
        if reference_type == "character":
            self.character_reference_image = processed_image
            print(f"🎭 Set character reference image")
        elif reference_type == "style":
            self.style_reference_image = processed_image
            print(f"🎨 Set style reference image")
        elif reference_type == "face":
            self.face_reference_image = processed_image
            print(f"👤 Set face reference image")
        
        return processed_image
    
    def _preprocess_reference_image(self, image):
        """Preprocess reference image for IP-Adapter"""
        # Resize to appropriate size (512x512 or 1024x1024)
        target_size = (512, 512)  # Adjust based on your model
        
        # Maintain aspect ratio while resizing
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste centered
        processed = Image.new("RGB", target_size, (255, 255, 255))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        processed.paste(image, (paste_x, paste_y))
        
        return processed
    
    def generate_with_character_consistency(self, prompt, negative_prompt="", **kwargs):
        """Generate image with character consistency using IP-Adapter"""
        if not self.available or self.character_reference_image is None:
            print("⚠️ IP-Adapter not available or no character reference set")
            return None
        
        try:
            # Set IP-Adapter image and weight
            self.base_pipeline.set_ip_adapter_scale(self.config["character_weight"])
            
            # Generate with IP-Adapter
            result = self.base_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image=self.character_reference_image,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            print(f"❌ IP-Adapter generation failed: {e}")
            if self.config["fallback_to_clip"]:
                print("🔄 Falling back to standard generation")
                return self.base_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    **kwargs
                )
            return None
    
    def generate_batch_with_consistency(self, prompt, negative_prompt="", num_images=3, **kwargs):
        """Generate multiple images with character consistency"""
        if not self.available:
            return None
        
        # Update kwargs for batch generation
        generation_kwargs = {
            "num_images_per_prompt": num_images,
            **kwargs
        }
        
        return self.generate_with_character_consistency(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **generation_kwargs
        )
    
    def update_character_reference_from_best(self, selected_image):
        """Update character reference with the best generated image"""
        if selected_image is not None:
            self.character_reference_image = self._preprocess_reference_image(selected_image)
            print("🔄 Updated character reference with best generated image")
    
    def extract_face_from_reference(self, reference_image):
        """Extract face region from reference image for face-specific consistency"""
        try:
            import cv2
            
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
            
            # Load face detection model
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region with some padding
                padding = int(min(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(opencv_image.shape[1], x + w + padding)
                y2 = min(opencv_image.shape[0], y + h + padding)
                
                face_region = opencv_image[y1:y2, x1:x2]
                face_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                
                self.face_reference_image = self._preprocess_reference_image(face_image)
                print("👤 Extracted and set face reference")
                return face_image
            else:
                print("⚠️ No face detected in reference image")
                return None
                
        except ImportError:
            print("⚠️ OpenCV not available for face extraction")
            return None
        except Exception as e:
            print(f"⚠️ Face extraction failed: {e}")
            return None
    
    def get_consistency_score(self, generated_image):
        """Calculate consistency score between generated image and reference"""
        if not self.available or self.character_reference_image is None:
            return 0.5
        
        try:
            # Use CLIP to compare generated image with reference
            from transformers import CLIPProcessor, CLIPModel
            
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            if torch.cuda.is_available():
                clip_model = clip_model.to("cuda")
            
            # Get embeddings
            inputs = clip_processor(
                images=[self.character_reference_image, generated_image], 
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(
                    image_features[0:1], 
                    image_features[1:2], 
                    dim=1
                ).item()
            
            return float(similarity)
            
        except Exception as e:
            print(f"⚠️ Consistency scoring failed: {e}")
            return 0.5
    
    def update_config(self, **config_updates):
        """Update IP-Adapter configuration"""
        self.config.update(config_updates)
        print(f"🔧 Updated IP-Adapter config: {config_updates}")
    
    def get_status(self):
        """Get IP-Adapter status"""
        return {
            "available": self.available,
            "character_reference_set": self.character_reference_image is not None,
            "style_reference_set": self.style_reference_image is not None,
            "face_reference_set": self.face_reference_image is not None,
            "config": self.config
        }
    
    def reset_references(self):
        """Reset all reference images"""
        self.character_reference_image = None
        self.style_reference_image = None
        self.face_reference_image = None
        print("🔄 Reset all reference images")


# Installation helper
def install_ip_adapter_requirements():
    """Helper function to install IP-Adapter requirements"""
    install_commands = [
        "pip install diffusers[ip-adapter]",
        "pip install transformers accelerate",
        "pip install opencv-python",  # For face extraction
        # Alternative IP-Adapter implementations:
        # "pip install ip_adapter", 
        # "git clone https://github.com/tencent-ailab/IP-Adapter.git"
    ]
    
    print("🔧 To use IP-Adapter, run these commands:")
    for cmd in install_commands:
        print(f"   {cmd}")
    
    print("\n📁 You'll also need to download IP-Adapter weights:")
    print("   https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models")
    print("   Download: ip-adapter_sdxl.bin")


if __name__ == "__main__":
    print("🎨 IP-Adapter Manager")
    print("For character-consistent storyboard generation")
    install_ip_adapter_requirements()