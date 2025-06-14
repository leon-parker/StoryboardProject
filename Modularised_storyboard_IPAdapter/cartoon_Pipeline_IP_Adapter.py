"""
Cartoon Pipeline with IP-Adapter Integration
Main interface for character-consistent cartoon generation
"""

import os
import torch
import time
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image

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
        
        # Check model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            self.available = False
            return
        
        try:
            # Load main pipeline
            self._load_diffusion_pipeline()
            
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
            self.available = False
    
    def _default_config(self):
        """Default generation configuration"""
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
                "character_weight": 0.8,
                "update_reference_from_best": True,
                "fallback_to_clip": True
            }
        }
    
    def _load_diffusion_pipeline(self):
        """Load the Stable Diffusion XL pipeline"""
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        )
        
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        
        print("✅ Diffusion pipeline loaded")
    
    def set_character_reference_image(self, reference_image_path_or_image):
        """Set the character reference image for IP-Adapter consistency"""
        if self.ip_adapter_manager.available:
            reference_image = self.ip_adapter_manager.set_character_reference(
                reference_image_path_or_image, 
                reference_type="character"
            )
            print("🎭 Character reference set for IP-Adapter consistency")
            return reference_image
        else:
            print("⚠️ IP-Adapter not available - using CLIP consistency only")
            return None
    
    def generate_single_image(self, prompt, negative_prompt="", use_ip_adapter=None, **kwargs):
        """Generate single best image with character consistency"""
        if not self.available:
            return None
        
        # Determine whether to use IP-Adapter
        if use_ip_adapter is None:
            use_ip_adapter = (
                self.config["selection"]["use_ip_adapter"] and 
                self.ip_adapter_manager.available and 
                self.ip_adapter_manager.character_reference_image is not None
            )
        
        # Use batch generation and select best
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
        """Generate multiple images and select best using IP-Adapter + CLIP + TIFA"""
        if not self.available:
            return None
        
        print(f"🎨 Generating {num_images} images for intelligent selection...")
        
        # Determine generation method
        if use_ip_adapter is None:
            use_ip_adapter = (
                self.config["selection"]["use_ip_adapter"] and 
                self.ip_adapter_manager.available and 
                self.ip_adapter_manager.character_reference_image is not None
            )
        
        # Prepare generation settings
        settings = self._prepare_generation_settings(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            **kwargs
        )
        
        # Generate images
        start_time = time.time()
        
        if use_ip_adapter:
            print("🎭 Using IP-Adapter for character consistency")
            result = self.ip_adapter_manager.generate_batch_with_consistency(
                **settings
            )
            used_ip_adapter = True
        else:
            print("🔄 Using standard generation")
            result = self.pipe(**settings)
            used_ip_adapter = False
        
        if result is None:
            print("❌ Image generation failed")
            return None
        
        gen_time = time.time() - start_time
        
        # Evaluate all generated images
        evaluation_results = self.quality_evaluator.evaluate_batch(result.images, prompt)
        
        # Select best image using combined scoring
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
        
        # Update IP-Adapter reference if configured
        if (used_ip_adapter and 
            self.config["ip_adapter"]["update_reference_from_best"] and
            self.ip_adapter_manager.available):
            self.ip_adapter_manager.update_character_reference_from_best(best_result["image"])
        
        # Return comprehensive result
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
    
    def _prepare_generation_settings(self, prompt, negative_prompt, num_images, **kwargs):
        """Prepare settings for image generation"""
        settings = {
            'prompt': prompt,
            'negative_prompt': negative_prompt or "blurry, low quality, distorted, deformed",
            'num_images_per_prompt': num_images,
            **self.config["generation"],
            **kwargs  # Allow override of defaults
        }
        return settings
    
    def _select_best_image_with_ip_adapter(self, images, evaluations, prompt, use_ip_adapter):
        """Select best image using TIFA + CLIP + IP-Adapter consistency"""
        is_first_image = len(self.consistency_manager.selected_images_history) == 0
        
        best_index = 0
        best_combined_score = -1
        
        for idx, (image, evaluation) in enumerate(zip(images, evaluations)):
            tifa_score = evaluation["score"]
            
            # Calculate CLIP consistency if not first image
            clip_consistency = 0.5
            if not is_first_image and self.consistency_manager.available:
                image_embedding = self.consistency_manager.get_image_embedding(image)
                clip_consistency = self.consistency_manager.calculate_consistency_score(image_embedding)
            
            # Calculate IP-Adapter consistency if used
            ip_adapter_consistency = 0.5
            if use_ip_adapter and self.ip_adapter_manager.available:
                ip_adapter_consistency = self.ip_adapter_manager.get_consistency_score(image)
            
            # Combine all scores
            if use_ip_adapter:
                # IP-Adapter mode: IP-Adapter gets highest weight
                combined_score = (
                    tifa_score * self.config["selection"]["quality_weight"] +
                    clip_consistency * self.config["selection"]["consistency_weight"] +
                    ip_adapter_consistency * self.config["selection"]["ip_adapter_weight"]
                )
            else:
                # Standard mode: Quality and CLIP consistency
                combined_score = (
                    tifa_score * self.config["selection"]["quality_weight"] +
                    clip_consistency * (self.config["selection"]["consistency_weight"] + 
                                      self.config["selection"]["ip_adapter_weight"])
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
    
    def generate_storyboard_sequence(self, prompts_list, character_reference_image=None, 
                                   iterations_per_prompt=1, images_per_iteration=3):
        """Generate character-consistent storyboard sequence"""
        if not self.available:
            return None
        
        print(f"📚 Generating character-consistent storyboard with {len(prompts_list)} prompts")
        
        # Set character reference if provided
        if character_reference_image is not None:
            self.set_character_reference_image(character_reference_image)
        
        sequence_results = []
        
        for prompt_idx, prompt in enumerate(prompts_list):
            print(f"\n📖 PROMPT {prompt_idx + 1}/{len(prompts_list)}: {prompt}")
            
            # For first prompt, we might not have a character reference yet
            # Generate normally and use the best result as character reference
            if (prompt_idx == 0 and 
                character_reference_image is None and
                self.ip_adapter_manager.character_reference_image is None):
                
                print("🎭 First prompt - will establish character reference")
                result = self.generate_with_selection(
                    prompt=prompt,
                    num_images=images_per_iteration,
                    use_ip_adapter=False  # Can't use IP-Adapter without reference
                )
                
                # Set the best result as character reference
                if result and self.ip_adapter_manager.available:
                    self.ip_adapter_manager.set_character_reference(
                        result["best_image"], 
                        reference_type="character"
                    )
                    print("🎭 Established character reference from first generation")
            else:
                # Use IP-Adapter for subsequent prompts
                if iterations_per_prompt > 1:
                    result = self.generate_with_iterations(
                        prompt=prompt,
                        iterations=iterations_per_prompt,
                        images_per_iteration=images_per_iteration
                    )
                    if result:
                        final_result = {
                            "best_image": result["best_result"]["image"],
                            "best_score": result["best_result"]["score"],
                            "used_ip_adapter": True
                        }
                else:
                    final_result = self.generate_with_selection(
                        prompt=prompt,
                        num_images=images_per_iteration,
                        use_ip_adapter=True
                    )
                
                result = final_result
            
            if result:
                sequence_results.append({
                    "prompt": prompt,
                    "prompt_index": prompt_idx,
                    "best_image": result["best_image"],
                    "final_score": result["best_score"],
                    "used_ip_adapter": result.get("used_ip_adapter", False),
                    "iterations_used": iterations_per_prompt
                })
            
            # Show consistency progress
            if prompt_idx > 0:
                # CLIP consistency report
                if self.consistency_manager.available:
                    consistency_report = self.consistency_manager.get_consistency_report()
                    if consistency_report["available"]:
                        print(f"📊 CLIP Consistency: {consistency_report['consistency_grade']} "
                              f"(Avg: {consistency_report['average_consistency']:.3f})")
                
                # IP-Adapter status
                if self.ip_adapter_manager.available:
                    status = self.ip_adapter_manager.get_status()
                    print(f"🎭 IP-Adapter: {'Active' if status['character_reference_set'] else 'Inactive'}")
        
        # Final reports
        final_consistency = self.consistency_manager.get_consistency_report()
        ip_adapter_status = self.ip_adapter_manager.get_status()
        
        print(f"\n🎉 STORYBOARD SEQUENCE COMPLETE!")
        print(f"📖 Generated: {len(sequence_results)} images")
        print(f"🎭 IP-Adapter used: {sum(1 for r in sequence_results if r.get('used_ip_adapter', False))} times")
        
        if final_consistency["available"]:
            print(f"📊 Overall CLIP Consistency: {final_consistency['consistency_grade']} "
                  f"(Score: {final_consistency['average_consistency']:.3f})")
        
        return {
            "sequence_results": sequence_results,
            "consistency_report": final_consistency,
            "ip_adapter_status": ip_adapter_status,
            "total_prompts": len(prompts_list),
            "successful_generations": len(sequence_results),
            "character_reference_established": ip_adapter_status["character_reference_set"]
        }
    
    def generate_with_iterations(self, prompt, iterations=3, images_per_iteration=3):
        """Generate with iterative prompt improvement using IP-Adapter"""
        print(f"🔄 Starting iterative improvement with IP-Adapter...")
        
        from prompt_improver import PromptImprover
        prompt_improver = PromptImprover(self.consistency_manager, self.quality_evaluator)
        
        current_prompt = prompt
        current_negative = "blurry, low quality, distorted, deformed"
        results_history = []
        
        for iteration in range(iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            
            result = self.generate_with_selection(
                prompt=current_prompt,
                negative_prompt=current_negative,
                num_images=images_per_iteration,
                use_ip_adapter=True  # Use IP-Adapter for consistency
            )
            
            if result is None:
                break
            
            # Analyze and improve (same as before)
            is_first_selected = len(self.consistency_manager.selected_images_history) == 1
            analysis = prompt_improver.analyze_prompt_image_alignment(
                result["best_image"], 
                current_prompt,
                is_first_image=is_first_selected
            )
            
            iteration_result = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "image": result["best_image"],
                "score": result["best_score"],
                "used_ip_adapter": result.get("used_ip_adapter", False),
                "clip_similarity": analysis["clip_similarity"],
                "caption": analysis["caption"]
            }
            
            # Improve prompts for next iteration
            if iteration < iterations - 1:
                improved = prompt_improver.improve_prompts(current_prompt, analysis, current_negative)
                current_prompt = improved["positive"]
                current_negative = improved["negative"]
                iteration_result["improvements_applied"] = improved["improvements"]
            
            results_history.append(iteration_result)
        
        if not results_history:
            return None
        
        best_iteration = max(results_history, key=lambda x: x["score"])
        
        return {
            "best_result": best_iteration,
            "history": results_history,
            "initial_prompt": prompt,
            "score_improvement": best_iteration["score"] - results_history[0]["score"]
        }
    
    def get_pipeline_status(self):
        """Get comprehensive pipeline status"""
        base_status = {
            "pipeline_available": self.available,
            "consistency_manager": self.consistency_manager.available,
            "quality_evaluator": self.quality_evaluator.available,
            "selected_images_count": len(self.consistency_manager.selected_images_history),
            "model_path": self.model_path
        }
        
        # Add IP-Adapter status
        if hasattr(self, 'ip_adapter_manager'):
            base_status.update({
                "ip_adapter": self.ip_adapter_manager.get_status()
            })
        
        return base_status
    
    def reset_all_memory(self):
        """Reset all consistency memory and references"""
        self.consistency_manager.reset_memory()
        if hasattr(self, 'ip_adapter_manager'):
            self.ip_adapter_manager.reset_references()
        print("🔄 Reset all memory - ready for new storyboard")
    
    def update_config(self, **config_updates):
        """Update pipeline configuration"""
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        # Update IP-Adapter config if present
        if "ip_adapter" in config_updates and hasattr(self, 'ip_adapter_manager'):
            self.ip_adapter_manager.update_config(**config_updates["ip_adapter"])
        
        print(f"🔧 Configuration updated: {config_updates}")


# Example usage
if __name__ == "__main__":
    print("🎨 Cartoon Pipeline with IP-Adapter - Character Consistent Storyboards")
    
    # Initialize pipeline
    pipeline = CartoonPipelineWithIPAdapter(
        model_path="path/to/realcartoonxl_v7.safetensors",
        ip_adapter_path="path/to/ip-adapter_sdxl.bin"  # Download from HuggingFace
    )
    
    if pipeline.available:
        # Method 1: Set character reference first
        character_ref = pipeline.set_character_reference_image("character_reference.jpg")
        
        # Generate sequence with character consistency
        morning_routine = [
            "boy waking up in bed, cartoon style",
            "same boy brushing teeth in bathroom, cartoon style", 
            "same boy getting dressed in bedroom, cartoon style",
            "same boy eating breakfast at kitchen table, cartoon style"
        ]
        
        sequence = pipeline.generate_storyboard_sequence(
            prompts_list=morning_routine,
            iterations_per_prompt=2,
            images_per_iteration=4
        )
        
        if sequence:
            print(f"✅ Generated {len(sequence['sequence_results'])} consistent images")
            print(f"🎭 Character reference: {sequence['character_reference_established']}")
            
            # Save images
            for i, result in enumerate(sequence['sequence_results']):
                result['best_image'].save(f"storyboard_frame_{i+1}.png")
                print(f"💾 Saved frame {i+1}: {result['prompt'][:50]}...")
        
        # Method 2: Let first image establish character reference
        sequence2 = pipeline.generate_storyboard_sequence(
            prompts_list=morning_routine,
            character_reference_image=None  # Will use first generated image
        )
        
        # Get status
        status = pipeline.get_pipeline_status()
        print(f"📊 Pipeline Status: {status}")
        
    else:
        print("❌ Pipeline not available")
        print("\n🔧 To use IP-Adapter, you need:")
        print("1. Install: pip install diffusers[ip-adapter]")
        print("2. Download IP-Adapter weights from HuggingFace")
        print("3. Ensure your model file exists")