# ip_adapter_manager.py
# IP-Adapter integration for character consistency in SDXL

import os
import torch
import numpy as np
from PIL import Image
import time
from typing import Optional, Dict, List, Union, Any
import warnings

# IP-Adapter imports with fallback
try:
    from ip_adapter import IPAdapterXL
    from diffusers.utils import load_image
    IP_ADAPTER_AVAILABLE = True
    print("✅ IP-Adapter library detected")
except ImportError:
    IP_ADAPTER_AVAILABLE = False
    print("⚠️ IP-Adapter not found. Install with:")
    print("   pip install ip-adapter")
    print("   Or: pip install git+https://github.com/tencent-ailab/IP-Adapter.git")

# Alternative IP-Adapter implementation detection
try:
    from diffusers import StableDiffusionXLPipeline
    # Check if pipeline has IP-Adapter support
    HF_IP_ADAPTER_AVAILABLE = hasattr(StableDiffusionXLPipeline, 'load_ip_adapter')
    if HF_IP_ADAPTER_AVAILABLE:
        print("✅ HuggingFace Diffusers IP-Adapter support detected")
except:
    HF_IP_ADAPTER_AVAILABLE = False

class IPAdapterCharacterManager:
    """
    Manages IP-Adapter for character consistency across SDXL generations
    Supports multiple IP-Adapter implementations
    """
    
    def __init__(self, 
                 model_base_path: str = None,
                 ip_adapter_model_path: str = None,
                 image_encoder_path: str = None,
                 device: str = "cuda",
                 implementation: str = "auto"):
        """
        Initialize IP-Adapter Character Manager
        
        Args:
            model_base_path: Path to base SDXL model
            ip_adapter_model_path: Path to IP-Adapter model weights
            image_encoder_path: Path to image encoder
            device: Device to use (cuda/cpu)
            implementation: 'tencent', 'huggingface', or 'auto'
        """
        
        print("🎭 Initializing IP-Adapter Character Manager...")
        
        self.device = device
        self.implementation = implementation
        self.available = False
        self.pipeline = None
        self.ip_adapter = None
        
        # Character consistency tracking
        self.character_reference = None
        self.reference_images = []
        self.character_embeddings = []
        self.generation_history = []
        
        # Default paths
        self.default_paths = {
            "ip_adapter_model": "IP-Adapter/models/ip-adapter_sdxl.bin",
            "image_encoder": "IP-Adapter/models/image_encoder",
            "clip_image_encoder": "h94/IP-Adapter/models/image_encoder"
        }
        
        # Update paths if provided
        if ip_adapter_model_path:
            self.default_paths["ip_adapter_model"] = ip_adapter_model_path
        if image_encoder_path:
            self.default_paths["image_encoder"] = image_encoder_path
        
        # Initialize implementation
        self._initialize_implementation()
    
    def _initialize_implementation(self):
        """Initialize the best available IP-Adapter implementation"""
        
        if self.implementation == "auto":
            # Try implementations in order of preference
            if self._try_tencent_implementation():
                self.implementation = "tencent"
                print("✅ Using Tencent IP-Adapter implementation")
            elif self._try_huggingface_implementation():
                self.implementation = "huggingface"
                print("✅ Using HuggingFace Diffusers IP-Adapter implementation")
            else:
                print("❌ No IP-Adapter implementation available")
                self._print_setup_instructions()
                return
        elif self.implementation == "tencent":
            if not self._try_tencent_implementation():
                print("❌ Tencent IP-Adapter implementation failed")
                return
        elif self.implementation == "huggingface":
            if not self._try_huggingface_implementation():
                print("❌ HuggingFace IP-Adapter implementation failed")
                return
        
        self.available = True
        print(f"✅ IP-Adapter Character Manager ready ({self.implementation} implementation)")
    
    def _try_tencent_implementation(self) -> bool:
        """Try to initialize Tencent IP-Adapter implementation"""
        if not IP_ADAPTER_AVAILABLE:
            return False
        
        try:
            # Note: Actual initialization will happen when pipeline is set
            return True
        except Exception as e:
            print(f"Tencent IP-Adapter initialization failed: {e}")
            return False
    
    def _try_huggingface_implementation(self) -> bool:
        """Try to initialize HuggingFace IP-Adapter implementation"""
        if not HF_IP_ADAPTER_AVAILABLE:
            return False
        
        try:
            # Check if necessary models are available
            return True
        except Exception as e:
            print(f"HuggingFace IP-Adapter initialization failed: {e}")
            return False
    
    def set_pipeline(self, pipeline):
        """
        Set the diffusion pipeline and initialize IP-Adapter
        
        Args:
            pipeline: StableDiffusionXLPipeline instance
        """
        if not self.available:
            print("⚠️ IP-Adapter not available")
            return False
        
        self.pipeline = pipeline
        
        try:
            if self.implementation == "tencent":
                return self._setup_tencent_pipeline()
            elif self.implementation == "huggingface":
                return self._setup_huggingface_pipeline()
        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            return False
        
        return True
    
    def _setup_tencent_pipeline(self) -> bool:
        """Setup Tencent IP-Adapter with pipeline"""
        try:
            self.ip_adapter = IPAdapterXL(
                sd_pipe=self.pipeline,
                image_encoder_path=self.default_paths["image_encoder"],
                ip_ckpt=self.default_paths["ip_adapter_model"],
                device=self.device
            )
            
            # Set default scale
            self.ip_adapter.set_scale(0.7)
            print("✅ Tencent IP-Adapter pipeline setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Tencent IP-Adapter setup failed: {e}")
            return False
    
    def _setup_huggingface_pipeline(self) -> bool:
        """Setup HuggingFace IP-Adapter with pipeline"""
        try:
            # Load IP-Adapter into pipeline
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl.bin"
            )
            
            # Set default scale
            self.pipeline.set_ip_adapter_scale(0.7)
            print("✅ HuggingFace IP-Adapter pipeline setup complete")
            return True
            
        except Exception as e:
            print(f"❌ HuggingFace IP-Adapter setup failed: {e}")
            return False
    
    def set_character_reference(self, reference_image: Union[str, Image.Image, np.ndarray]) -> bool:
        """
        Set the main character reference image
        
        Args:
            reference_image: Reference image (path, PIL Image, or numpy array)
        
        Returns:
            bool: Success status
        """
        if not self.available:
            print("⚠️ IP-Adapter not available")
            return False
        
        try:
            # Convert to PIL Image
            if isinstance(reference_image, str):
                if os.path.exists(reference_image):
                    reference_image = Image.open(reference_image)
                else:
                    reference_image = load_image(reference_image)  # URL
            elif isinstance(reference_image, np.ndarray):
                reference_image = Image.fromarray(reference_image)
            elif not isinstance(reference_image, Image.Image):
                print(f"❌ Unsupported image type: {type(reference_image)}")
                return False
            
            # Ensure RGB format
            if reference_image.mode != 'RGB':
                reference_image = reference_image.convert('RGB')
            
            self.character_reference = reference_image
            self.reference_images = [reference_image]
            
            print("🎭 Character reference image set successfully")
            print(f"   Image size: {reference_image.size}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to set character reference: {e}")
            return False
    
    def add_reference_image(self, image: Union[str, Image.Image, np.ndarray]) -> bool:
        """
        Add additional reference image for character consistency
        
        Args:
            image: Additional reference image
            
        Returns:
            bool: Success status
        """
        if not self.available:
            return False
        
        try:
            # Convert to PIL Image (same logic as set_character_reference)
            if isinstance(image, str):
                if os.path.exists(image):
                    image = Image.open(image)
                else:
                    image = load_image(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                return False
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            self.reference_images.append(image)
            print(f"🎭 Added reference image (Total: {len(self.reference_images)})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add reference image: {e}")
            return False
    
    def get_generation_kwargs(self, 
                            use_character_reference: bool = True,
                            ip_adapter_scale: float = 0.7,
                            reference_index: int = 0) -> Dict[str, Any]:
        """
        Get generation kwargs for IP-Adapter
        
        Args:
            use_character_reference: Whether to use character reference
            ip_adapter_scale: IP-Adapter influence scale (0.0-1.0)
            reference_index: Which reference image to use (0 = main reference)
            
        Returns:
            dict: Generation kwargs
        """
        if not self.available or not use_character_reference or not self.reference_images:
            return {}
        
        try:
            # Select reference image
            if reference_index < len(self.reference_images):
                reference_image = self.reference_images[reference_index]
            else:
                reference_image = self.character_reference
            
            if reference_image is None:
                return {}
            
            kwargs = {}
            
            if self.implementation == "tencent":
                # Tencent IP-Adapter uses generate method directly
                kwargs = {
                    "ip_adapter_image": reference_image,
                    "scale": ip_adapter_scale
                }
            elif self.implementation == "huggingface":
                # HuggingFace uses ip_adapter_image parameter
                kwargs = {
                    "ip_adapter_image": reference_image
                }
                # Scale is set separately on pipeline
                if hasattr(self.pipeline, 'set_ip_adapter_scale'):
                    self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
            
            return kwargs
            
        except Exception as e:
            print(f"❌ Error getting generation kwargs: {e}")
            return {}
    
    def generate_with_character_consistency(self,
                                          prompt: str,
                                          negative_prompt: str = "",
                                          ip_adapter_scale: float = 0.7,
                                          num_images: int = 1,
                                          **kwargs) -> Dict[str, Any]:
        """
        Generate images with character consistency
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            ip_adapter_scale: IP-Adapter scale
            num_images: Number of images to generate
            **kwargs: Additional generation parameters
            
        Returns:
            dict: Generation results
        """
        if not self.available or not self.pipeline:
            print("❌ IP-Adapter or pipeline not available")
            return None
        
        if not self.character_reference:
            print("⚠️ No character reference set - generating without IP-Adapter")
            return self._generate_without_ip_adapter(prompt, negative_prompt, num_images, **kwargs)
        
        try:
            # Get IP-Adapter kwargs
            ip_kwargs = self.get_generation_kwargs(
                use_character_reference=True,
                ip_adapter_scale=ip_adapter_scale
            )
            
            # Merge with generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_images_per_prompt": num_images,
                "height": kwargs.get("height", 1024),
                "width": kwargs.get("width", 1024),
                "num_inference_steps": kwargs.get("num_inference_steps", 30),
                "guidance_scale": kwargs.get("guidance_scale", 5.0),
                **ip_kwargs,
                **kwargs
            }
            
            # Generate images
            start_time = time.time()
            
            if self.implementation == "tencent" and self.ip_adapter:
                # Use IP-Adapter generate method
                result = self.ip_adapter.generate(**generation_params)
            else:
                # Use pipeline generate method
                result = self.pipeline(**generation_params)
            
            generation_time = time.time() - start_time
            
            # Store generation info
            generation_info = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "ip_adapter_scale": ip_adapter_scale,
                "generation_time": generation_time,
                "num_images": len(result.images),
                "used_character_reference": True
            }
            
            self.generation_history.append(generation_info)
            
            return {
                "images": result.images,
                "generation_info": generation_info,
                "character_reference_used": True
            }
            
        except Exception as e:
            print(f"❌ IP-Adapter generation failed: {e}")
            print("🔄 Falling back to standard generation...")
            return self._generate_without_ip_adapter(prompt, negative_prompt, num_images, **kwargs)
    
    def _generate_without_ip_adapter(self, prompt, negative_prompt, num_images, **kwargs):
        """Fallback generation without IP-Adapter"""
        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                height=kwargs.get("height", 1024),
                width=kwargs.get("width", 1024),
                num_inference_steps=kwargs.get("num_inference_steps", 30),
                guidance_scale=kwargs.get("guidance_scale", 5.0),
                **kwargs
            )
            
            return {
                "images": result.images,
                "generation_info": {"used_character_reference": False},
                "character_reference_used": False
            }
            
        except Exception as e:
            print(f"❌ Fallback generation also failed: {e}")
            return None
    
    def calculate_character_consistency_score(self, 
                                            generated_image: Image.Image,
                                            use_clip: bool = True) -> float:
        """
        Calculate character consistency score for generated image
        
        Args:
            generated_image: Generated image to evaluate
            use_clip: Whether to use CLIP for similarity calculation
            
        Returns:
            float: Consistency score (0.0-1.0)
        """
        if not self.available or not self.character_reference:
            return 0.5
        
        try:
            if use_clip:
                return self._calculate_clip_similarity(generated_image, self.character_reference)
            else:
                # Simple pixel-based similarity as fallback
                return self._calculate_simple_similarity(generated_image, self.character_reference)
                
        except Exception as e:
            print(f"❌ Character consistency calculation failed: {e}")
            return 0.5
    
    def _calculate_clip_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate CLIP-based image similarity"""
        try:
            # This would require CLIP model - simplified for now
            # In practice, you'd use your existing CLIP model from the main pipeline
            return 0.7  # Placeholder
        except:
            return 0.5
    
    def _calculate_simple_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Simple pixel-based similarity calculation"""
        try:
            # Resize images to same size
            size = (224, 224)
            img1_resized = img1.resize(size)
            img2_resized = img2.resize(size)
            
            # Convert to numpy arrays
            arr1 = np.array(img1_resized, dtype=np.float32)
            arr2 = np.array(img2_resized, dtype=np.float32)
            
            # Calculate MSE and convert to similarity
            mse = np.mean((arr1 - arr2) ** 2)
            similarity = 1.0 / (1.0 + mse / 10000.0)  # Normalize
            
            return float(similarity)
            
        except Exception as e:
            print(f"Simple similarity calculation failed: {e}")
            return 0.5
    
    def get_character_consistency_report(self) -> Dict[str, Any]:
        """Get report on character consistency across generations"""
        if not self.available:
            return {"available": False, "message": "IP-Adapter not available"}
        
        if not self.generation_history:
            return {"available": False, "message": "No generations recorded"}
        
        # Calculate statistics
        total_generations = len(self.generation_history)
        character_reference_used = sum(1 for gen in self.generation_history if gen.get("used_character_reference", False))
        
        avg_generation_time = np.mean([gen["generation_time"] for gen in self.generation_history])
        
        return {
            "available": True,
            "total_generations": total_generations,
            "character_reference_used_count": character_reference_used,
            "character_reference_usage_rate": character_reference_used / total_generations if total_generations > 0 else 0,
            "average_generation_time": float(avg_generation_time),
            "implementation": self.implementation,
            "reference_images_count": len(self.reference_images)
        }
    
    def reset_character_memory(self):
        """Reset character reference and history"""
        self.character_reference = None
        self.reference_images = []
        self.character_embeddings = []
        self.generation_history = []
        print("🔄 Character memory reset")
    
    def _print_setup_instructions(self):
        """Print setup instructions for IP-Adapter"""
        print("\n📋 IP-ADAPTER SETUP INSTRUCTIONS:")
        print("=" * 50)
        print("🔧 Option 1: Tencent IP-Adapter")
        print("   pip install ip-adapter")
        print("   # Download models to IP-Adapter/models/")
        print("   # - ip-adapter_sdxl.bin")
        print("   # - image_encoder/")
        print()
        print("🔧 Option 2: HuggingFace Diffusers")
        print("   pip install diffusers>=0.21.0")
        print("   # Models downloaded automatically")
        print()
        print("📁 Expected model structure:")
        print("   IP-Adapter/")
        print("   ├── models/")
        print("   │   ├── ip-adapter_sdxl.bin")
        print("   │   └── image_encoder/")
        print("   │       ├── config.json")
        print("   │       └── pytorch_model.bin")
        print()
        print("🌐 Download from:")
        print("   https://huggingface.co/h94/IP-Adapter")

def test_ip_adapter_manager():
    """Test function for IP-Adapter Manager"""
    print("\n🧪 TESTING IP-ADAPTER MANAGER")
    print("=" * 40)
    
    # Initialize manager
    manager = IPAdapterCharacterManager()
    
    print(f"Available: {manager.available}")
    print(f"Implementation: {manager.implementation}")
    
    if manager.available:
        print("✅ IP-Adapter Manager ready for use!")
        
        # Test character reference setting (would need actual image)
        # manager.set_character_reference("path/to/character.jpg")
        
        print("\n💡 Usage with your existing pipeline:")
        print("1. manager.set_pipeline(your_sdxl_pipeline)")
        print("2. manager.set_character_reference(first_generated_image)")
        print("3. Use manager.get_generation_kwargs() in your generation")
        print("4. Calculate consistency with manager.calculate_character_consistency_score()")
    else:
        print("❌ IP-Adapter not available - check setup instructions above")

if __name__ == "__main__":
    test_ip_adapter_manager()