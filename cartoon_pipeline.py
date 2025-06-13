"""
Cartoon Pipeline - Main interface for high-quality cartoon generation
Modular architecture ready for IP-Adapter integration
"""

import os
import torch
import time
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from consistency_manager import ConsistencyManager
from quality_evaluator import QualityEvaluator


class CartoonPipeline:
    """Main pipeline for consistent, high-quality cartoon generation"""
    
    def __init__(self, model_path, ip_adapter_path=None, config=None):
        print("🎨 Loading Cartoon Pipeline...")
        
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
            
            # TODO: Initialize IP-Adapter when ready
            self.ip_adapter_available = False
            if ip_adapter_path and os.path.exists(ip_adapter_path):
                self._load_ip_adapter()
            
            self.available = True
            print("✅ Cartoon Pipeline ready!")
            
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
                "consistency_weight": 0.5,
                "quality_weight": 0.5
            },
            "ip_adapter": {
                "enabled": False,
                "face_adapter_weight": 0.8,
                "style_adapter_weight": 0.7,
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
    
    def _load_ip_adapter(self):
        """Load IP-Adapter (placeholder for future implementation)"""
        # TODO: Implement IP-Adapter loading
        print("🔄 IP-Adapter loading not yet implemented")
        self.ip_adapter_available = False
    
    def generate_single_image(self, prompt, negative_prompt="", **kwargs):
        """Generate single best image with quality evaluation"""
        if not self.available:
            return None
        
        # Use batch generation and select best
        result = self.generate_with_selection(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=self.config["generation"]["num_images_per_prompt"],
            **kwargs
        )
        
        if result:
            return {
                "image": result["best_image"],
                "score": result["best_score"],
                "generation_time": result["generation_time"]
            }
        return None
    
    def generate_with_selection(self, prompt, negative_prompt="", num_images=3, use_consistency=None, **kwargs):
        """Generate multiple images and select best based on quality and consistency"""
        if not self.available:
            return None
        
        print(f"🎨 Generating {num_images} images for intelligent selection...")
        
        # Determine if we should use consistency
        if use_consistency is None:
            use_consistency = self.config["selection"]["use_consistency"]
        
        # Prepare generation settings
        settings = self._prepare_generation_settings(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images=num_images,
            **kwargs
        )
        
        # Generate images
        start_time = time.time()
        
        if self.ip_adapter_available and self.config["ip_adapter"]["enabled"]:
            result = self._generate_with_ip_adapter(settings)
        else:
            result = self.pipe(**settings)
        
        gen_time = time.time() - start_time
        
        # Evaluate all generated images
        evaluation_results = self.quality_evaluator.evaluate_batch(result.images, prompt)
        
        # Select best image based on quality and consistency
        best_result = self._select_best_image(
            images=result.images,
            evaluations=evaluation_results,
            prompt=prompt,
            use_consistency=use_consistency
        )
        
        # Store selected image for consistency tracking
        if self.consistency_manager.available and best_result:
            self.consistency_manager.store_selected_image(
                image=best_result["image"],
                prompt=prompt,
                quality_score=best_result["score"]
            )
        
        # Return comprehensive result
        return {
            "best_image": best_result["image"],
            "best_score": best_result["score"],
            "best_index": best_result["index"],
            "all_images": result.images,
            "all_evaluations": evaluation_results,
            "generation_time": gen_time,
            "selection_method": "ip_adapter" if self.ip_adapter_available else "clip_tifa",
            "consistency_used": use_consistency and len(self.consistency_manager.selected_images_history) > 0
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
    
    def _generate_with_ip_adapter(self, settings):
        """Generate images using IP-Adapter (placeholder)"""
        # TODO: Implement IP-Adapter generation
        # For now, fallback to regular generation
        print("🔄 IP-Adapter generation not yet implemented, using standard pipeline")
        return self.pipe(**settings)
    
    def _select_best_image(self, images, evaluations, prompt, use_consistency):
        """Select best image from batch using quality and consistency scores"""
        is_first_image = len(self.consistency_manager.selected_images_history) == 0
        
        best_index = 0
        best_combined_score = -1
        
        for idx, (image, evaluation) in enumerate(zip(images, evaluations)):
            quality_score = evaluation["score"]
            
            # Calculate consistency if not first image and consistency is enabled
            if use_consistency and not is_first_image and self.consistency_manager.available:
                image_embedding = self.consistency_manager.get_image_embedding(image)
                consistency_score = self.consistency_manager.calculate_consistency_score(image_embedding)
                
                # Combine quality and consistency
                combined_score = (
                    quality_score * self.config["selection"]["quality_weight"] +
                    consistency_score * self.config["selection"]["consistency_weight"]
                )
            else:
                combined_score = quality_score
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_index = idx
        
        return {
            "image": images[best_index],
            "score": best_combined_score,
            "index": best_index,
            "evaluation": evaluations[best_index]
        }
    
    def generate_sequence(self, prompts_list, iterations_per_prompt=1, images_per_iteration=3):
        """Generate a consistent sequence of images"""
        if not self.available:
            return None
        
        print(f"📚 Generating sequence with {len(prompts_list)} prompts")
        print(f"🔄 {iterations_per_prompt} iterations per prompt, {images_per_iteration} images per iteration")
        
        sequence_results = []
        
        for prompt_idx, prompt in enumerate(prompts_list):
            print(f"\n📖 PROMPT {prompt_idx + 1}/{len(prompts_list)}: {prompt}")
            
            if iterations_per_prompt > 1:
                # Use iterative improvement
                result = self.generate_with_iterations(
                    prompt=prompt,
                    iterations=iterations_per_prompt,
                    images_per_iteration=images_per_iteration
                )
                if result:
                    sequence_results.append({
                        "prompt": prompt,
                        "prompt_index": prompt_idx,
                        "best_image": result["best_result"]["image"],
                        "final_score": result["best_result"]["score"],
                        "iterations_used": iterations_per_prompt
                    })
            else:
                # Single generation
                result = self.generate_with_selection(
                    prompt=prompt,
                    num_images=images_per_iteration
                )
                if result:
                    sequence_results.append({
                        "prompt": prompt,
                        "prompt_index": prompt_idx,
                        "best_image": result["best_image"],
                        "final_score": result["best_score"],
                        "iterations_used": 1
                    })
            
            # Show consistency progress
            if prompt_idx > 0 and self.consistency_manager.available:
                consistency_report = self.consistency_manager.get_consistency_report()
                if consistency_report["available"]:
                    print(f"📊 Sequence Consistency: {consistency_report['consistency_grade']} "
                          f"(Avg: {consistency_report['average_consistency']:.3f})")
        
        # Final consistency report
        final_consistency = self.consistency_manager.get_consistency_report()
        
        print(f"\n🎉 SEQUENCE COMPLETE!")
        print(f"📖 Generated: {len(sequence_results)} images")
        
        if final_consistency["available"]:
            print(f"📊 Overall Consistency: {final_consistency['consistency_grade']} "
                  f"(Score: {final_consistency['average_consistency']:.3f})")
        
        return {
            "sequence_results": sequence_results,
            "consistency_report": final_consistency,
            "total_prompts": len(prompts_list),
            "successful_generations": len(sequence_results)
        }
    
    def generate_with_iterations(self, prompt, iterations=3, images_per_iteration=3):
        """Generate with iterative prompt improvement"""
        print(f"🔄 Starting iterative improvement generation...")
        print(f"📝 Initial prompt: {prompt}")
        
        # Initialize prompt improver
        from prompt_improver import PromptImprover
        prompt_improver = PromptImprover(self.consistency_manager, self.quality_evaluator)
        
        current_prompt = prompt
        current_negative = "blurry, low quality, distorted, deformed"
        
        results_history = []
        
        for iteration in range(iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            print(f"📝 Current prompt: {current_prompt}")
            print(f"🚫 Current negative: {current_negative}")
            
            # Generate images with current prompts
            result = self.generate_with_selection(
                prompt=current_prompt,
                negative_prompt=current_negative,
                num_images=images_per_iteration
            )
            
            if result is None:
                break
            
            # Analyze the best image with consistency checking
            is_first_selected = len(self.consistency_manager.selected_images_history) == 1  # Just stored this image
            analysis = prompt_improver.analyze_prompt_image_alignment(
                result["best_image"], 
                current_prompt,
                is_first_image=is_first_selected
            )
            
            print(f"📊 CLIP similarity: {analysis['clip_similarity']:.3f}")
            print(f"📝 Generated caption: {analysis['caption']}")
            print(f"🎯 Alignment quality: {analysis['alignment_quality']}")
            
            # Show consistency issues if detected
            consistency_issues = analysis.get("consistency_issues", {})
            if consistency_issues["severity"] != "low":
                print(f"⚠️ Consistency issues ({consistency_issues['severity']}): {len(consistency_issues['character_inconsistencies'])} character, {len(consistency_issues['artifacts_detected'])} artifacts")
            else:
                print("✅ No major consistency issues detected")
            
            # Store iteration result
            iteration_result = {
                "iteration": iteration + 1,
                "prompt": current_prompt,
                "negative_prompt": current_negative,
                "image": result["best_image"],
                "score": result["best_score"],
                "clip_similarity": analysis["clip_similarity"],
                "caption": analysis["caption"],
                "improvements_applied": []
            }
            
            # Improve prompts for next iteration (if not last iteration)
            if iteration < iterations - 1:
                improved = prompt_improver.improve_prompts(
                    current_prompt, 
                    analysis, 
                    current_negative
                )
                
                current_prompt = improved["positive"]
                current_negative = improved["negative"]
                iteration_result["improvements_applied"] = improved["improvements"]
                
                print(f"🚀 Applied improvements: {', '.join(improved['improvements'])}")
            
            results_history.append(iteration_result)
        
        if not results_history:
            print("❌ No successful iterations")
            return None
        
        # Find best result across all iterations
        best_iteration = max(results_history, key=lambda x: x["score"])
        
        print(f"\n🏆 ITERATIVE IMPROVEMENT COMPLETE")
        print(f"📈 Score progression: {[f'{r['score']:.3f}' for r in results_history]}")
        print(f"🥇 Best iteration: {best_iteration['iteration']} (Score: {best_iteration['score']:.3f})")
        print(f"📊 Best CLIP similarity: {best_iteration['clip_similarity']:.3f}")
        
        return {
            "best_result": best_iteration,
            "history": results_history,
            "initial_prompt": prompt,
            "final_prompt": results_history[-1]["prompt"],
            "score_improvement": best_iteration["score"] - results_history[0]["score"]
        }
    
    def set_reference_image(self, reference_image_path):
        """Set reference image for IP-Adapter consistency"""
        if not self.ip_adapter_available:
            print("⚠️ IP-Adapter not available, storing for CLIP consistency only")
        
        # TODO: Implement IP-Adapter reference setting
        print(f"🎨 Reference image setting not yet implemented: {reference_image_path}")
    
    def update_config(self, **config_updates):
        """Update pipeline configuration"""
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        print(f"🔧 Configuration updated: {config_updates}")
    
    def get_consistency_report(self):
        """Get detailed consistency report"""
        return self.consistency_manager.get_consistency_report()
    
    def reset_consistency_memory(self):
        """Reset consistency tracking (for new sequence)"""
        self.consistency_manager.reset_memory()
        print("🔄 Consistency memory reset - ready for new sequence")
    
    def get_pipeline_status(self):
        """Get status of all pipeline components"""
        return {
            "pipeline_available": self.available,
            "consistency_manager": self.consistency_manager.available,
            "quality_evaluator": self.quality_evaluator.available,
            "ip_adapter": self.ip_adapter_available,
            "selected_images_count": len(self.consistency_manager.selected_images_history),
            "model_path": self.model_path,
            "ip_adapter_path": self.ip_adapter_path
        }


# Example usage
if __name__ == "__main__":
    print("🎨 Cartoon Pipeline - Modular Example")
    
    # Initialize pipeline
    pipeline = CartoonPipeline(
        model_path="path/to/realcartoonxl_v7.safetensors",
        ip_adapter_path="path/to/ip-adapter.bin"  # Optional
    )
    
    if pipeline.available:
        # Single image generation
        result = pipeline.generate_single_image("boy brushing teeth")
        
        # Sequence generation
        prompts = ["boy waking up", "boy brushing teeth", "boy eating breakfast"]
        sequence = pipeline.generate_sequence(prompts)
        
        # Get consistency report
        report = pipeline.get_consistency_report()
        print(f"Consistency: {report}")
    else:
        print("❌ Pipeline not available")