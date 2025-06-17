"""
Cartoon Pipeline with IP-Adapter Integration - BAKED VAE COMPATIBILITY FIXED
Fixes black image issues with baked VAE models like RealCartoon XL
"""

import os
import torch
import time
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Import real modular components instead of placeholder classes
from consistency_manager import ConsistencyManager
from quality_evaluator import QualityEvaluator  
from ip_adapter_manager import IPAdapterManager

class CartoonPipelineWithIPAdapter:
    """Main pipeline for character-consistent cartoon generation using IP-Adapter"""
    
    def __init__(self, model_path, ip_adapter_path=None, config=None):
        print("🎨 Loading Cartoon Pipeline with IP-Adapter...")
        
        self.model_path = model_path
        self.ip_adapter_path = ip_adapter_path
        self.config = config or self._default_config()
        
        # Initialize character reference tracking
        self.character_reference_image = None
        self.ip_adapter_loaded = False
        self.baked_vae_compatible = False
        
        # Check model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            self.available = False
            return
        
        try:
            # Load main pipeline
            self._load_diffusion_pipeline()
            
            # CRITICAL: Test baked VAE compatibility first
            self._test_baked_vae_compatibility()
            
            # CRITICAL: ALWAYS load IP-Adapter - no conditional loading
            self._load_ip_adapter_mandatory()
            
            # CRITICAL: Fix tensor dtype issues for baked VAE
            self._fix_tensor_dtypes_baked_vae()
            
            # Test IP-Adapter with baked VAE
            self._test_ip_adapter_baked_vae_compatibility()
            
            # Initialize evaluation components
            self.consistency_manager = ConsistencyManager()
            self.quality_evaluator = QualityEvaluator()
            
            # Initialize IP-Adapter manager
            self.ip_adapter_manager = IPAdapterManager(
                base_pipeline=self.pipe,
                ip_adapter_path=ip_adapter_path
            )
            
            self.available = True
            print("✅ Cartoon Pipeline with IP-Adapter ready!")
            
        except Exception as e:
            print(f"❌ Pipeline loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.available = False
    
    def _default_config(self):
        """Default generation configuration optimized for baked VAE"""
        return {
            "generation": {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 5.0,
                "num_images_per_prompt": 3
            },
            "selection": {
                "use_consistency": True,
                "use_ip_adapter": True,
                "quality_weight": 0.4,
                "consistency_weight": 0.3,
                "ip_adapter_weight": 0.3
            },
            "ip_adapter": {
                "character_weight": 0.3,  # LOWERED for baked VAE compatibility
                "update_reference_from_best": True,
                "fallback_to_clip": False
            },
            "baked_vae": {
                "use_conservative_optimizations": True,
                "disable_autocast_on_failure": True,
                "force_float32_fallback": True
            }
        }
    
    def _load_diffusion_pipeline(self):
        """Load pipeline optimized for baked VAE models"""
        print("🔧 Loading SDXL pipeline with baked VAE support...")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True,
            variant="fp16"
        )
        
        self.pipe = self.pipe.to("cuda")
        
        # Conservative VAE optimizations for baked models
        try:
            # Try VAE tiling first (safer for baked VAE)
            if hasattr(self.pipe, 'enable_vae_tiling'):
                self.pipe.enable_vae_tiling()
                print("✅ VAE tiling enabled for baked VAE")
            else:
                # Fallback to slicing with caution
                self.pipe.enable_vae_slicing()
                print("✅ VAE slicing enabled (fallback)")
        except Exception as e:
            print(f"⚠️ VAE optimization skipped: {e}")
        
        print("✅ Baked VAE pipeline loaded")
    
    def _test_baked_vae_compatibility(self):
        """Test if baked VAE works correctly"""
        print("🧪 Testing baked VAE compatibility...")
        
        try:
            # Check VAE properties
            if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                print(f"🔍 VAE type: {type(self.pipe.vae).__name__}")
                print(f"🔍 VAE device: {self.pipe.vae.device}")
                print(f"🔍 VAE dtype: {self.pipe.vae.dtype}")
                
                if hasattr(self.pipe.vae, 'config'):
                    scaling_factor = getattr(self.pipe.vae.config, 'scaling_factor', 'unknown')
                    print(f"🔍 VAE scaling factor: {scaling_factor}")
                    
                    if scaling_factor != 0.13025:
                        print(f"⚠️ Non-standard VAE scaling: {scaling_factor}")
            
            # Simple test generation
            test_prompt = "a simple red circle on white background"
            print("🧪 Testing basic generation...")
            
            result = self.pipe(
                prompt=test_prompt,
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512,
                guidance_scale=3.0
            )
            
            # Check if result is black
            test_image = result.images[0]
            img_array = np.array(test_image)
            
            # More sophisticated black image detection
            mean_value = np.mean(img_array)
            std_value = np.std(img_array)
            
            if mean_value < 15 and std_value < 10:  # Very dark with low variation
                print("❌ Baked VAE producing black/dark images")
                self.baked_vae_compatible = False
                
                # Try with different settings
                print("🔧 Trying VAE fallback settings...")
                return self._try_vae_fallback_test()
            else:
                print(f"✅ Baked VAE working (mean: {mean_value:.1f}, std: {std_value:.1f})")
                self.baked_vae_compatible = True
                return True
                
        except Exception as e:
            print(f"❌ Baked VAE test failed: {e}")
            self.baked_vae_compatible = False
            return False
    
    def _try_vae_fallback_test(self):
        """Try alternative settings for problematic baked VAE"""
        print("🔧 Trying VAE fallback configurations...")
        
        try:
            # Disable VAE optimizations
            if hasattr(self.pipe, 'disable_vae_slicing'):
                self.pipe.disable_vae_slicing()
            if hasattr(self.pipe, 'disable_vae_tiling'):
                self.pipe.disable_vae_tiling()
            
            # Test with float32 VAE
            original_vae_dtype = self.pipe.vae.dtype
            self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
            
            result = self.pipe(
                prompt="a simple red circle on white background",
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512,
                guidance_scale=3.0
            )
            
            test_image = result.images[0]
            img_array = np.array(test_image)
            mean_value = np.mean(img_array)
            
            # Restore VAE dtype
            self.pipe.vae = self.pipe.vae.to(dtype=original_vae_dtype)
            
            if mean_value > 15:
                print("✅ VAE fallback successful - float32 fixes the issue")
                self.config["baked_vae"]["force_float32_fallback"] = True
                self.baked_vae_compatible = True
                return True
            else:
                print("❌ VAE fallback still produces black images")
                return False
                
        except Exception as e:
            print(f"❌ VAE fallback test failed: {e}")
            return False
    
    def _load_ip_adapter_mandatory(self):
        """Load IP-Adapter with baked VAE considerations"""
        print("🎭 Loading IP-Adapter for baked VAE model...")
        
        try:
            print("📥 Loading IP-Adapter from HuggingFace...")
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", 
                subfolder="sdxl_models", 
                weight_name="ip-adapter_sdxl.bin",
                torch_dtype=torch.float16
            )
            
            # Set conservative initial scale for baked VAE
            initial_scale = self.config["ip_adapter"]["character_weight"]
            self.pipe.set_ip_adapter_scale(initial_scale)
            self.ip_adapter_loaded = True
            print(f"✅ IP-Adapter loaded with scale {initial_scale}")
            
        except Exception as e:
            print(f"❌ HuggingFace IP-Adapter failed: {e}")
            
            # Try local file if provided
            if self.ip_adapter_path and os.path.exists(self.ip_adapter_path):
                try:
                    print(f"📥 Trying local IP-Adapter: {self.ip_adapter_path}")
                    self.pipe.load_ip_adapter(self.ip_adapter_path, torch_dtype=torch.float16)
                    self.pipe.set_ip_adapter_scale(self.config["ip_adapter"]["character_weight"])
                    self.ip_adapter_loaded = True
                    print("✅ IP-Adapter loaded from local file")
                except Exception as e2:
                    print(f"❌ Local IP-Adapter failed: {e2}")
                    raise Exception(f"IP-Adapter loading failed: {e2}")
            else:
                raise Exception(f"IP-Adapter loading failed and no local backup: {e}")
    
    def _fix_tensor_dtypes_baked_vae(self):
        """Fix tensor dtypes specifically for baked VAE + IP-Adapter"""
        print("🔧 Fixing tensor dtypes for baked VAE + IP-Adapter...")
        
        try:
            # Handle baked VAE specifically
            if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                print(f"🔍 Baked VAE device: {self.pipe.vae.device}")
                print(f"🔍 Baked VAE dtype: {self.pipe.vae.dtype}")
                
                # Ensure VAE is on CUDA with correct dtype
                self.pipe.vae = self.pipe.vae.to(device="cuda", dtype=torch.float16)
                print("✅ Baked VAE synced to CUDA float16")
            
            # IP-Adapter components
            if hasattr(self.pipe, 'image_encoder') and self.pipe.image_encoder is not None:
                self.pipe.image_encoder = self.pipe.image_encoder.to(device="cuda", dtype=torch.float16)
                print("✅ Image encoder synced with baked VAE")
            
            # UNet
            if hasattr(self.pipe, 'unet'):
                self.pipe.unet = self.pipe.unet.to(device="cuda", dtype=torch.float16)
                print("✅ UNet synced")
            
            # Text encoders
            if hasattr(self.pipe, 'text_encoder'):
                self.pipe.text_encoder = self.pipe.text_encoder.to(device="cuda", dtype=torch.float16)
            if hasattr(self.pipe, 'text_encoder_2'):
                self.pipe.text_encoder_2 = self.pipe.text_encoder_2.to(device="cuda", dtype=torch.float16)
            
            torch.cuda.synchronize()
            print("✅ Baked VAE + IP-Adapter tensor sync complete")
            
        except Exception as e:
            print(f"⚠️ Tensor sync warning: {e}")
    
    def _test_ip_adapter_baked_vae_compatibility(self):
        """Test IP-Adapter with baked VAE"""
        print("🧪 Testing IP-Adapter with baked VAE...")
        
        if not self.ip_adapter_loaded:
            print("⚠️ IP-Adapter not loaded, skipping compatibility test")
            return
        
        try:
            # Create simple test image
            dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
            
            # Test with minimal IP-Adapter influence
            self.pipe.set_ip_adapter_scale(0.1)
            
            result = self.pipe(
                prompt="a simple blue square",
                ip_adapter_image=dummy_image,
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512,
                guidance_scale=3.0
            )
            
            test_image = result.images[0]
            img_array = np.array(test_image)
            mean_value = np.mean(img_array)
            
            if mean_value < 15:
                print("❌ IP-Adapter + baked VAE producing black images")
                # Lower the default IP-Adapter scale even more
                self.config["ip_adapter"]["character_weight"] = 0.15
                print("🔧 Lowered IP-Adapter scale to 0.15")
            else:
                print(f"✅ IP-Adapter + baked VAE working (mean: {mean_value:.1f})")
            
        except Exception as e:
            print(f"❌ IP-Adapter + baked VAE test failed: {e}")
            # Emergency fallback - very low scale
            self.config["ip_adapter"]["character_weight"] = 0.05
            print("🚨 Emergency fallback: IP-Adapter scale set to 0.05")
    
    def _create_proper_dummy_image(self):
        """Create a simple dummy image for IP-Adapter"""
        # Simple gray image - no complex preprocessing
        return Image.new('RGB', (1024, 1024), color=(128, 128, 128))
    
    def set_character_reference_image(self, reference_image_path_or_image):
        """Set the character reference image for IP-Adapter consistency"""
        if isinstance(reference_image_path_or_image, str):
            reference_image = Image.open(reference_image_path_or_image).convert("RGB")
        else:
            reference_image = reference_image_path_or_image
        
        # Resize to consistent size
        if reference_image.size != (1024, 1024):
            reference_image = reference_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        self.character_reference_image = reference_image
        print("🎭 Character reference set for IP-Adapter consistency")
        return reference_image
    
    def generate_single_image(self, prompt, negative_prompt="", use_ip_adapter=None, **kwargs):
        """Generate single best image with character consistency"""
        if not self.available:
            return None
        
        # Use IP-Adapter if loaded and reference exists
        if use_ip_adapter is None:
            use_ip_adapter = (
                self.ip_adapter_loaded and 
                self.character_reference_image is not None
            )
        
        result = self.generate_with_selection(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=self.config["generation"]["num_images_per_prompt"],
            use_ip_adapter=use_ip_adapter,
            **kwargs
        )
        
        if result:
            return {
                "image": result["best_image"],
                "score": result["best_score"],
                "generation_time": result["generation_time"],
                "used_ip_adapter": result.get("used_ip_adapter", False)
            }
        return None
    
    def generate_with_selection(self, prompt, negative_prompt="", num_images=3, use_ip_adapter=None, **kwargs):
        """Generate multiple images with baked VAE optimizations"""
        if not self.available:
            return None
        
        print(f"🎨 Generating {num_images} images for intelligent selection...")
        
        # Force IP-Adapter if loaded
        if use_ip_adapter is None:
            use_ip_adapter = self.ip_adapter_loaded
        
        # Prepare generation settings
        settings = self._prepare_generation_settings(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            **kwargs
        )
        
        start_time = time.time()
        
        # ===== BAKED VAE + IP-ADAPTER IMPLEMENTATION =====
        if use_ip_adapter and self.ip_adapter_loaded:
            if self.character_reference_image is not None:
                print("🎭 Using IP-Adapter with character reference")
                
                # Set IP-Adapter scale (lower for baked VAE)
                self.pipe.set_ip_adapter_scale(self.config["ip_adapter"]["character_weight"])
                
                # Try generation with baked VAE error handling
                result = self._generate_with_baked_vae_handling(
                    ip_adapter_image=self.character_reference_image,
                    settings=settings
                )
                used_ip_adapter = True
                
            else:
                # First image - use minimal IP-Adapter influence
                print("🎭 First image - minimal IP-Adapter influence")
                
                dummy_reference = self._create_proper_dummy_image()
                self.pipe.set_ip_adapter_scale(0.05)  # Very minimal for first image
                
                result = self._generate_with_baked_vae_handling(
                    ip_adapter_image=dummy_reference,
                    settings=settings
                )
                used_ip_adapter = True
        else:
            # Fallback without IP-Adapter
            print("🎨 Generating without IP-Adapter")
            result = self._generate_with_baked_vae_handling(settings=settings)
            used_ip_adapter = False
        
        if result is None or not result.images:
            raise Exception("Image generation failed - check baked VAE compatibility!")
        
        # Check for black images
        for i, img in enumerate(result.images):
            img_array = np.array(img)
            mean_val = np.mean(img_array)
            if mean_val < 15:
                print(f"⚠️ Image {i+1} appears very dark (mean: {mean_val:.1f})")
        
        gen_time = time.time() - start_time
        
        # Evaluate all generated images
        evaluation_results = self.quality_evaluator.evaluate_batch(result.images, prompt)
        
        # Select best image
        best_result = self._select_best_image_with_ip_adapter(
            images=result.images,
            evaluations=evaluation_results,
            prompt=prompt,
            use_ip_adapter=used_ip_adapter
        )
        
        # Store selected image for consistency tracking
        if self.consistency_manager.available and best_result:
            self.consistency_manager.store_selected_image(
                image=best_result["image"],
                prompt=prompt,
                tifa_score=best_result["tifa_score"]
            )
        
        # Update character reference from first generation
        if (self.character_reference_image is None and best_result and 
            self.config["ip_adapter"]["update_reference_from_best"]):
            print("🎭 Setting first generated image as character reference")
            self.set_character_reference_image(best_result["image"])
            # Increase IP-Adapter scale for subsequent images
            self.pipe.set_ip_adapter_scale(self.config["ip_adapter"]["character_weight"])
        
        return {
            "best_image": best_result["image"],
            "best_score": best_result["combined_score"],
            "best_index": best_result["index"],
            "all_images": result.images,
            "all_evaluations": evaluation_results,
            "generation_time": gen_time,
            "used_ip_adapter": used_ip_adapter,
            "consistency_used": len(self.consistency_manager.selected_images_history) > 0
        }
    
    def _generate_with_baked_vae_handling(self, ip_adapter_image=None, settings=None):
        """Generate with proper baked VAE error handling"""
        try:
            # First attempt - normal generation
            if ip_adapter_image is not None:
                result = self.pipe(ip_adapter_image=ip_adapter_image, **settings)
            else:
                result = self.pipe(**settings)
            
            # Quick black image check
            if result and result.images:
                first_img = np.array(result.images[0])
                if np.mean(first_img) > 15:  # Not black
                    return result
                else:
                    print("⚠️ Black image detected, trying fallback...")
            
        except Exception as e:
            print(f"⚠️ Generation failed: {e}, trying fallback...")
        
        # Fallback generation for baked VAE issues
        return self._baked_vae_fallback_generation(ip_adapter_image, settings)
    
    def _baked_vae_fallback_generation(self, ip_adapter_image, settings):
        """Fallback generation for baked VAE issues"""
        print("🔧 Using baked VAE fallback generation...")
        
        try:
            # Disable VAE optimizations
            if hasattr(self.pipe, 'disable_vae_slicing'):
                self.pipe.disable_vae_slicing()
            if hasattr(self.pipe, 'disable_vae_tiling'):
                self.pipe.disable_vae_tiling()
            
            # Try with float32 VAE if enabled in config
            if self.config["baked_vae"]["force_float32_fallback"]:
                original_vae_dtype = self.pipe.vae.dtype
                self.pipe.vae = self.pipe.vae.to(dtype=torch.float32)
                
                try:
                    if ip_adapter_image is not None:
                        # Lower IP-Adapter scale for fallback
                        original_scale = self.pipe.get_ip_adapter_scale() if hasattr(self.pipe, 'get_ip_adapter_scale') else None
                        self.pipe.set_ip_adapter_scale(0.1)
                        result = self.pipe(ip_adapter_image=ip_adapter_image, **settings)
                        if original_scale is not None:
                            self.pipe.set_ip_adapter_scale(original_scale)
                    else:
                        result = self.pipe(**settings)
                    
                    # Restore VAE dtype
                    self.pipe.vae = self.pipe.vae.to(dtype=original_vae_dtype)
                    
                    return result
                    
                except Exception as e:
                    # Restore VAE dtype even on failure
                    self.pipe.vae = self.pipe.vae.to(dtype=original_vae_dtype)
                    print(f"❌ Float32 fallback failed: {e}")
            
            # Final attempt without IP-Adapter
            if ip_adapter_image is not None:
                print("🚨 Final fallback: disabling IP-Adapter")
                return self.pipe(**settings)
            
            return None
            
        except Exception as e:
            print(f"❌ All fallback attempts failed: {e}")
            return None
        finally:
            # Re-enable optimizations
            try:
                if hasattr(self.pipe, 'enable_vae_tiling'):
                    self.pipe.enable_vae_tiling()
                elif hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()
            except:
                pass
    
    def _prepare_generation_settings(self, prompt, negative_prompt, num_images, **kwargs):
        """Prepare settings for image generation"""
        settings = {
            'prompt': prompt,
            'negative_prompt': negative_prompt or "blurry, low quality, distorted, deformed",
            'num_images_per_prompt': num_images,
            **self.config["generation"],
            **kwargs
        }
        
        return settings
    
    def _select_best_image_with_ip_adapter(self, images, evaluations, prompt, use_ip_adapter):
        """Select best image using TIFA + CLIP + IP-Adapter consistency"""
        is_first_image = len(self.consistency_manager.selected_images_history) == 0
        
        best_index = 0
        best_combined_score = -1
        
        for idx, (image, evaluation) in enumerate(zip(images, evaluations)):
            tifa_score = evaluation["score"]
            
            # Check for black image and penalize
            img_array = np.array(image)
            mean_brightness = np.mean(img_array)
            brightness_penalty = 0
            
            if mean_brightness < 15:  # Very dark image
                brightness_penalty = 0.5  # Heavy penalty
                print(f"⚠️ Image {idx} is very dark (brightness: {mean_brightness:.1f})")
            elif mean_brightness < 30:  # Somewhat dark
                brightness_penalty = 0.2  # Moderate penalty
            
            # Calculate CLIP consistency if not first image
            clip_consistency = 0.5
            if not is_first_image and self.consistency_manager.available:
                image_embedding = self.consistency_manager.get_image_embedding(image)
                clip_consistency = self.consistency_manager.calculate_consistency_score(image_embedding)
            
            # Calculate IP-Adapter consistency if used
            ip_adapter_consistency = 0.5
            if use_ip_adapter and self.ip_adapter_manager and self.ip_adapter_manager.available:
                ip_adapter_consistency = self.ip_adapter_manager.get_consistency_score(image)
            
            # Combine all scores with brightness penalty
            if use_ip_adapter:
                combined_score = (
                    tifa_score * self.config["selection"]["quality_weight"] +
                    clip_consistency * self.config["selection"]["consistency_weight"] +
                    ip_adapter_consistency * self.config["selection"]["ip_adapter_weight"] -
                    brightness_penalty
                )
            else:
                combined_score = (
                    tifa_score * self.config["selection"]["quality_weight"] +
                    clip_consistency * (self.config["selection"]["consistency_weight"] + 
                                      self.config["selection"]["ip_adapter_weight"]) -
                    brightness_penalty
                )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_index = idx
        
        return {
            "image": images[best_index],
            "combined_score": best_combined_score,
            "tifa_score": evaluations[best_index]["score"],
            "index": best_index,
            "evaluation": evaluations[best_index]
        }
    
    def generate_with_progressive_improvement(self, prompt, num_images=3, use_ip_adapter=None, 
                                             quality_threshold=0.5, max_iterations=3):
        """Generate images with iterative prompt improvement using existing PromptImprover"""
        from prompt_improver import PromptImprover
        
        prompt_improver = PromptImprover(self.consistency_manager, self.quality_evaluator)
        
        current_prompt = prompt
        original_prompt = prompt
        current_negative = "blurry, low quality, distorted, deformed"
        
        best_image = None
        best_score = 0.0
        score_progression = []
        prompt_history = [prompt]
        progressive_improvement_applied = False
        
        for iteration in range(max_iterations):
            print(f"🔄 Iteration {iteration + 1}/{max_iterations}: {current_prompt[:60]}...")
            
            # Generate images using your existing selection method
            result = self.generate_with_selection(
                prompt=current_prompt,
                negative_prompt=current_negative,
                num_images=num_images,
                use_ip_adapter=use_ip_adapter
            )
            
            if not result:
                print(f"⚠️ Generation failed in iteration {iteration + 1}")
                break
            
            current_score = result["best_score"]
            current_image = result["best_image"]
            score_progression.append(current_score)
            
            # Update overall best if this iteration is better
            if current_score > best_score:
                best_score = current_score
                best_image = current_image
            
            print(f"✨ Score this iteration: {current_score:.3f}")
            
            # Analyze the result using your PromptImprover
            is_first_selected = len(self.consistency_manager.selected_images_history) == 1
            analysis = prompt_improver.analyze_prompt_image_alignment(
                current_image, 
                current_prompt,
                is_first_image=is_first_selected
            )
            
            print(f"📊 CLIP similarity: {analysis['clip_similarity']:.3f}")
            print(f"📝 Caption: {analysis['caption'][:60]}...")
            
            # Check consistency issues
            consistency_issues = analysis.get("consistency_issues", {})
            if consistency_issues["severity"] != "low":
                print(f"⚠️ Consistency issues detected: {consistency_issues['severity']}")
            
            # Check if we should continue improving
            if current_score >= quality_threshold:
                print(f"✅ Quality threshold ({quality_threshold}) reached!")
                break
            
            # Only try to improve prompt if we haven't reached the last iteration
            if iteration < max_iterations - 1:
                # Generate improved prompts using your PromptImprover
                improved = prompt_improver.improve_prompts(
                    current_prompt, 
                    analysis, 
                    current_negative
                )
                
                new_prompt = improved["positive"]
                new_negative = improved["negative"]
                
                if new_prompt != current_prompt:
                    current_prompt = new_prompt
                    current_negative = new_negative
                    prompt_history.append(current_prompt)
                    progressive_improvement_applied = True
                    
                    print(f"🔧 Prompt refined: {current_prompt[:60]}...")
                    if improved["improvements"]:
                        print(f"✨ Applied improvements: {', '.join(improved['improvements'][:2])}...")
                else:
                    print("🤔 No prompt improvement suggested")
        
        # Show final improvement summary
        if len(score_progression) > 1:
            total_improvement = score_progression[-1] - score_progression[0]
            print(f"📈 Score progression: {score_progression[0]:.3f} → {score_progression[-1]:.3f} (Δ{total_improvement:+.3f})")
        
        return {
            "best_image": best_image,
            "best_score": best_score,
            "used_ip_adapter": use_ip_adapter and hasattr(self, 'ip_adapter_manager') and self.ip_adapter_manager.available,
            "progressive_improvement": progressive_improvement_applied,
            "final_prompt": current_prompt,
            "original_prompt": original_prompt,
            "score_progression": score_progression,
            "prompt_history": prompt_history,
            "iterations_completed": len(score_progression)
        }
    
    def generate_storyboard_sequence(self, prompts_list, character_reference_image=None, 
                                    iterations_per_prompt=1, images_per_iteration=3):
        """Generate character-consistent storyboard sequence with progressive prompt improvement"""
        if not self.available:
            return None
        
        # Check IP-Adapter availability properly
        ip_adapter_available = hasattr(self, 'ip_adapter_manager') and self.ip_adapter_manager.available
        if not ip_adapter_available:
            print("⚠️ IP-Adapter not loaded - storyboard may lack consistency")
        
        print(f"📚 Generating storyboard with {len(prompts_list)} prompts with progressive prompt improvement")
        
        # Set character reference if provided
        if character_reference_image is not None:
            self.set_character_reference_image(character_reference_image)
        
        sequence_results = []
        character_reference_established = character_reference_image is not None
        ip_adapter_used_count = 0
        
        for prompt_idx, prompt in enumerate(prompts_list):
            print(f"\n📖 PROMPT {prompt_idx + 1}/{len(prompts_list)}: {prompt}")
            
            # Use progressive improvement for each prompt
            result = self.generate_with_progressive_improvement(
                prompt=prompt,
                num_images=images_per_iteration,
                use_ip_adapter=ip_adapter_available,
                quality_threshold=0.5  # Adjust threshold as needed
            )
            
            if not result:
                print(f"⚠️ Generation failed for prompt {prompt_idx + 1}")
                continue
            
            used_ip_adapter = result.get("used_ip_adapter", False)
            if used_ip_adapter:
                ip_adapter_used_count += 1
            
            # Enhanced result tracking with progressive improvement data
            sequence_results.append({
                "prompt": prompt,
                "prompt_index": prompt_idx,
                "best_image": result["best_image"],
                "final_score": result["best_score"],
                "used_ip_adapter": used_ip_adapter,
                "iterations_used": iterations_per_prompt,
                "progressive_improvement": result.get("progressive_improvement", False),
                "final_prompt": result.get("final_prompt", prompt),
                "original_prompt": result.get("original_prompt", prompt),
                "score_progression": result.get("score_progression", [result["best_score"]]),
                "prompt_history": result.get("prompt_history", [prompt])
            })
            
            # Show improvement details
            if result.get("progressive_improvement", False):
                scores = result.get("score_progression", [])
                if len(scores) > 1:
                    improvement = scores[-1] - scores[0]
                    print(f"📈 Progressive improvement: {scores[0]:.3f} → {scores[-1]:.3f} (Δ{improvement:+.3f})")
                
                # Show final prompt if it changed
                final_prompt = result.get("final_prompt", prompt)
                if final_prompt != prompt:
                    print(f"✨ Final optimized prompt: {final_prompt[:80]}...")
            
            # Mark character reference as established after first generation
            if prompt_idx == 0 and hasattr(self, 'character_reference_image') and self.character_reference_image is not None:
                character_reference_established = True
            
            # Show progress
            if prompt_idx > 0 and hasattr(self, 'consistency_manager') and self.consistency_manager.available:
                consistency_report = self.consistency_manager.get_consistency_report()
                if consistency_report["available"]:
                    print(f"📊 CLIP Consistency: {consistency_report['consistency_grade']} "
                          f"(Avg: {consistency_report['average_consistency']:.3f})")
        
        # Final reports
        final_consistency = {"available": False}
        if hasattr(self, 'consistency_manager') and self.consistency_manager.available:
            final_consistency = self.consistency_manager.get_consistency_report()
        
        # Calculate improvement statistics
        total_improvements = sum(1 for r in sequence_results if r.get("progressive_improvement", False))
        total_score_improvement = sum(
            (max(r["score_progression"]) - min(r["score_progression"])) 
            for r in sequence_results if len(r.get("score_progression", [])) > 1
        )
        
        print(f"\n🎉 STORYBOARD SEQUENCE COMPLETE!")
        print(f"📖 Generated: {len(sequence_results)} images")
        print(f"🎭 IP-Adapter used: {ip_adapter_used_count} times")
        print(f"🔧 Progressive improvement applied: {total_improvements}/{len(sequence_results)} prompts")
        if total_score_improvement > 0:
            print(f"📈 Total quality improvement: +{total_score_improvement:.3f}")
        
        # Handle baked VAE status if available
        baked_vae_compatible = getattr(self, 'baked_vae_compatible', True)
        baked_vae_config = getattr(self, 'config', {}).get("baked_vae", {})
        print(f"🔧 Baked VAE compatible: {baked_vae_compatible}")
        
        if final_consistency["available"]:
            print(f"📊 Overall CLIP Consistency: {final_consistency['consistency_grade']} "
                  f"(Score: {final_consistency['average_consistency']:.3f})")
        
        return {
            "sequence_results": sequence_results,
            "consistency_report": final_consistency,
            "ip_adapter_status": {
                "available": ip_adapter_available,
                "character_reference_set": hasattr(self, 'character_reference_image') and self.character_reference_image is not None,
                "used_count": ip_adapter_used_count
            },
            "progressive_improvement_stats": {
                "prompts_improved": total_improvements,
                "total_prompts": len(sequence_results),
                "total_score_improvement": total_score_improvement,
                "improvement_rate": total_improvements / len(sequence_results) if sequence_results else 0
            },
            "baked_vae_status": {
                "compatible": baked_vae_compatible,
                "optimizations_used": baked_vae_config
            },
            "total_prompts": len(prompts_list),
            "successful_generations": len(sequence_results),
            "character_reference_established": character_reference_established,
            "total_frames": len(sequence_results),
            "ip_adapter_used_count": ip_adapter_used_count
        }
    
    def get_pipeline_status(self):
        """Get comprehensive pipeline status including baked VAE info"""
        base_status = {
            "pipeline_available": self.available,
            "consistency_manager": self.consistency_manager.available if hasattr(self, 'consistency_manager') else False,
            "quality_evaluator": self.quality_evaluator.available if hasattr(self, 'quality_evaluator') else False,
            "selected_images_count": len(self.consistency_manager.selected_images_history) if hasattr(self, 'consistency_manager') else 0,
            "model_path": self.model_path,
            "character_reference_set": self.character_reference_image is not None,
            "ip_adapter_loaded": self.ip_adapter_loaded,
            "baked_vae_compatible": self.baked_vae_compatible,
            "current_ip_adapter_scale": self.config["ip_adapter"]["character_weight"]
        }
        
        return base_status
    
    def reset_all_memory(self):
        """Reset all consistency memory and references"""
        if hasattr(self, 'consistency_manager'):
            self.consistency_manager.reset_memory()
        self.character_reference_image = None
        if hasattr(self, 'ip_adapter_manager') and self.ip_adapter_manager:
            self.ip_adapter_manager.reset_references()
        print("🔄 Reset all memory - ready for new storyboard")
    
    def update_config(self, **config_updates):
        """Update pipeline configuration with baked VAE considerations"""
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        # Update IP-Adapter scale if it changed
        if "ip_adapter" in config_updates and "character_weight" in config_updates["ip_adapter"]:
            if self.ip_adapter_loaded and hasattr(self.pipe, 'set_ip_adapter_scale'):
                new_scale = config_updates["ip_adapter"]["character_weight"]
                # Validate scale for baked VAE compatibility
                if not self.baked_vae_compatible and new_scale > 0.5:
                    print(f"⚠️ High IP-Adapter scale ({new_scale}) may cause issues with this baked VAE")
                self.pipe.set_ip_adapter_scale(new_scale)
        
        print(f"🔧 Configuration updated: {config_updates}")
    
    def diagnose_black_image_issues(self):
        """Diagnostic function for black image issues"""
        print("🔍 DIAGNOSING BLACK IMAGE ISSUES")
        print("=" * 40)
        
        # Check VAE status
        if hasattr(self.pipe, 'vae'):
            print(f"VAE Type: {type(self.pipe.vae).__name__}")
            print(f"VAE Device: {self.pipe.vae.device}")
            print(f"VAE Dtype: {self.pipe.vae.dtype}")
            
            if hasattr(self.pipe.vae, 'config'):
                scaling = getattr(self.pipe.vae.config, 'scaling_factor', 'unknown')
                print(f"VAE Scaling Factor: {scaling}")
        
        # Check IP-Adapter status
        print(f"IP-Adapter Loaded: {self.ip_adapter_loaded}")
        if self.ip_adapter_loaded:
            try:
                current_scale = self.pipe.get_ip_adapter_scale() if hasattr(self.pipe, 'get_ip_adapter_scale') else 'unknown'
                print(f"IP-Adapter Scale: {current_scale}")
            except:
                print(f"IP-Adapter Scale: {self.config['ip_adapter']['character_weight']} (config)")
        
        # Check baked VAE compatibility
        print(f"Baked VAE Compatible: {self.baked_vae_compatible}")
        
        # Memory status
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB used")
            print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.1f}GB")
        
        print("=" * 40)
        
        # Suggestions
        print("🔧 SUGGESTIONS:")
        if not self.baked_vae_compatible:
            print("- Try lowering IP-Adapter scale to 0.1 or lower")
            print("- Enable float32 VAE fallback in config")
            print("- Disable VAE optimizations")
        
        if self.ip_adapter_loaded:
            current_scale = self.config["ip_adapter"]["character_weight"]
            if current_scale > 0.3:
                print(f"- Current IP-Adapter scale ({current_scale}) may be too high for baked VAE")
                print("- Try scale between 0.1-0.3 for baked VAE models")
    
    def emergency_fix_black_images(self):
        """Emergency function to fix black image generation"""
        print("🚨 EMERGENCY BLACK IMAGE FIX")
        
        # Step 1: Lower IP-Adapter scale dramatically
        if self.ip_adapter_loaded:
            emergency_scale = 0.05
            self.pipe.set_ip_adapter_scale(emergency_scale)
            self.config["ip_adapter"]["character_weight"] = emergency_scale
            print(f"✅ Set IP-Adapter scale to emergency level: {emergency_scale}")
        
        # Step 2: Disable VAE optimizations
        try:
            if hasattr(self.pipe, 'disable_vae_slicing'):
                self.pipe.disable_vae_slicing()
            if hasattr(self.pipe, 'disable_vae_tiling'):
                self.pipe.disable_vae_tiling()
            print("✅ Disabled VAE optimizations")
        except:
            pass
        
        # Step 3: Enable float32 fallback
        self.config["baked_vae"]["force_float32_fallback"] = True
        print("✅ Enabled float32 VAE fallback")
        
        # Step 4: Clear GPU cache
        torch.cuda.empty_cache()
        print("✅ Cleared GPU cache")
        
        # Step 5: Test generation
        print("🧪 Testing emergency fix...")
        try:
            test_result = self.pipe(
                prompt="a simple red circle",
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512,
                guidance_scale=3.0
            )
            
            if test_result and test_result.images:
                test_img = np.array(test_result.images[0])
                mean_val = np.mean(test_img)
                
                if mean_val > 15:
                    print(f"✅ Emergency fix successful! (brightness: {mean_val:.1f})")
                    return True
                else:
                    print(f"❌ Still producing dark images (brightness: {mean_val:.1f})")
                    return False
            else:
                print("❌ No images generated")
                return False
                
        except Exception as e:
            print(f"❌ Emergency test failed: {e}")
            return False

# Enhanced test and reporting functions
def generate_enhanced_test_report(sequence_result, output_dir, total_time, pipeline_status=None):
    """Generate comprehensive test report with baked VAE diagnostics"""
    import os
    
    report_path = os.path.join(output_dir, "enhanced_test_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ENHANCED CARTOON PIPELINE TEST REPORT\n")
        f.write("Baked VAE + IP-Adapter Compatibility Analysis\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Generation Time: {total_time:.1f}s\n")
        f.write(f"Total Frames Generated: {sequence_result.get('total_frames', 0)}\n")
        f.write(f"Average Time per Frame: {total_time/max(sequence_result.get('total_frames', 1), 1):.1f}s\n\n")
        
        # Pipeline status
        if pipeline_status:
            f.write("PIPELINE STATUS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Pipeline Available: {pipeline_status.get('pipeline_available', False)}\n")
            f.write(f"Baked VAE Compatible: {pipeline_status.get('baked_vae_compatible', False)}\n")
            f.write(f"IP-Adapter Loaded: {pipeline_status.get('ip_adapter_loaded', False)}\n")
            f.write(f"Current IP-Adapter Scale: {pipeline_status.get('current_ip_adapter_scale', 'unknown')}\n\n")
        
        # Baked VAE specific analysis
        if 'baked_vae_status' in sequence_result:
            vae_status = sequence_result['baked_vae_status']
            f.write("BAKED VAE ANALYSIS:\n")
            f.write("-" * 18 + "\n")
            f.write(f"VAE Compatibility: {vae_status.get('compatible', 'Unknown')}\n")
            f.write(f"Optimizations Used: {vae_status.get('optimizations_used', {})}\n\n")
        
        # IP-Adapter analysis
        f.write("IP-ADAPTER ANALYSIS:\n")
        f.write("-" * 19 + "\n")
        f.write(f"Character Reference Established: {sequence_result.get('character_reference_established', False)}\n")
        f.write(f"IP-Adapter Usage Count: {sequence_result.get('ip_adapter_used_count', 0)}\n")
        f.write(f"Expected Usage Count: {sequence_result.get('total_frames', 0)}\n")
        
        usage_rate = 0
        if sequence_result.get('total_frames', 0) > 0:
            usage_rate = sequence_result.get('ip_adapter_used_count', 0) / sequence_result.get('total_frames', 1) * 100
        f.write(f"IP-Adapter Usage Rate: {usage_rate:.1f}%\n\n")
        
        # Test results
        f.write("TEST RESULTS:\n")
        f.write("-" * 12 + "\n")
        
        # Pass/Fail criteria
        tests = [
            ("Images Generated", sequence_result.get('total_frames', 0) > 0),
            ("IP-Adapter Used", sequence_result.get('ip_adapter_used_count', 0) > 0),
            ("Character Reference", sequence_result.get('character_reference_established', False)),
            ("Baked VAE Compatible", pipeline_status.get('baked_vae_compatible', True) if pipeline_status else True)
        ]
        
        for test_name, passed in tests:
            status = "✅ PASS" if passed else "❌ FAIL"
            f.write(f"{status}: {test_name}\n")
        
        # Consistency report
        if 'consistency_report' in sequence_result:
            consistency = sequence_result['consistency_report']
            if consistency.get('available', False):
                f.write(f"\nCONSISTENCY ANALYSIS:\n")
                f.write("-" * 19 + "\n")
                f.write(f"Consistency Grade: {consistency.get('consistency_grade', 'N/A')}\n")
                f.write(f"Average Consistency: {consistency.get('average_consistency', 0):.3f}\n")
        
        # Detailed frame analysis
        f.write("\n" + "=" * 60 + "\n")
        f.write("DETAILED FRAME ANALYSIS\n")
        f.write("=" * 60 + "\n")
        
        if 'sequence_results' in sequence_result:
            for i, frame in enumerate(sequence_result['sequence_results']):
                f.write(f"\nFrame {i+1}:\n")
                f.write(f"  Prompt: {frame.get('prompt', 'N/A')}\n")
                f.write(f"  IP-Adapter Used: {frame.get('used_ip_adapter', False)}\n")
                f.write(f"  Final Score: {frame.get('final_score', 0):.3f}\n")
                f.write(f"  Generation Status: {'Success' if frame.get('best_image') else 'Failed'}\n")
        
        # Recommendations
        f.write("\n" + "=" * 60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n")
        
        if pipeline_status and not pipeline_status.get('baked_vae_compatible', True):
            f.write("⚠️  BAKED VAE COMPATIBILITY ISSUES DETECTED:\n")
            f.write("   - Consider lowering IP-Adapter scale to 0.1-0.3\n")
            f.write("   - Enable float32 VAE fallback in config\n")
            f.write("   - Try disabling VAE optimizations\n\n")
        
        if usage_rate < 80:
            f.write("⚠️  LOW IP-ADAPTER USAGE DETECTED:\n")
            f.write("   - Check for tensor dtype conflicts\n")
            f.write("   - Verify IP-Adapter loading process\n")
            f.write("   - Consider emergency fix procedures\n\n")
        
        f.write("✅ For optimal results with baked VAE models:\n")
        f.write("   - Use IP-Adapter scales between 0.1-0.4\n")
        f.write("   - Enable conservative VAE optimizations\n")
        f.write("   - Monitor for black image generation\n")
        f.write("   - Use emergency_fix_black_images() if needed\n")
    
    print(f"📋 Enhanced test report saved to: {report_path}")

def run_baked_vae_diagnostics(pipeline):
    """Run comprehensive diagnostics for baked VAE compatibility"""
    print("\n🔬 RUNNING BAKED VAE DIAGNOSTICS")
    print("=" * 50)
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    
    # Run black image diagnosis
    pipeline.diagnose_black_image_issues()
    
    # Test with different IP-Adapter scales
    print("\n🧪 TESTING DIFFERENT IP-ADAPTER SCALES:")
    scales_to_test = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    for scale in scales_to_test:
        print(f"\nTesting scale {scale}...")
        
        # Update scale
        pipeline.update_config(ip_adapter={"character_weight": scale})
        
        try:
            # Quick test generation
            result = pipeline.generate_single_image(
                prompt="a simple cartoon character",
                num_images_per_prompt=1,
                num_inference_steps=10,
                height=512, width=512
            )
            
            if result and result["image"]:
                img_array = np.array(result["image"])
                mean_brightness = np.mean(img_array)
                
                if mean_brightness > 15:
                    print(f"  ✅ Scale {scale}: Working (brightness: {mean_brightness:.1f})")
                else:
                    print(f"  ❌ Scale {scale}: Black image (brightness: {mean_brightness:.1f})")
            else:
                print(f"  ❌ Scale {scale}: Generation failed")
                
        except Exception as e:
            print(f"  ❌ Scale {scale}: Error - {e}")
    
    print("\n" + "=" * 50)
    return status

if __name__ == "__main__":
    print("🎨 Enhanced Cartoon Pipeline with Baked VAE Support")
    print("✅ Fixes black image issues with RealCartoon XL and similar models")
    print("🎭 Optimized IP-Adapter integration for baked VAE compatibility")
    print("🔧 Includes diagnostic and emergency fix capabilities")
    print("\n🚀 Key features:")
    print("   - Automatic baked VAE compatibility testing")
    print("   - Smart IP-Adapter scale adjustment")
    print("   - Black image detection and prevention")
    print("   - Emergency fix procedures")
    print("   - Comprehensive diagnostics")