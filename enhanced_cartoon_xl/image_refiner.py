# EnhancedAIImageRefiner - Extracted directly from original code
from diffusers import StableDiffusionXLImg2ImgPipeline
import torch
import config
import os
import time

class EnhancedAIImageRefiner:
    """Enhanced AI-powered image refinement using img2img with validation-informed prompts"""
    
    def __init__(self, model_path):
        print("🔧 Loading Enhanced AI Image Refiner...")
        
        self.model_path = model_path
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            self.available = False
            return
        
        try:
            print("📁 Loading img2img refinement pipeline...")
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                model_path, torch_dtype=torch.float16, use_safetensors=True
            )
            
            self.refiner_pipe = self.refiner_pipe.to("cuda")
            self.refiner_pipe.enable_model_cpu_offload()
            self.refiner_pipe.enable_vae_slicing()
            
            self.available = True
            print("✅ Enhanced AI Image Refiner ready!")
            
        except Exception as e:
            print(f"❌ AI refiner loading failed: {e}")
            self.available = False
    
    def create_ai_refinement_prompt(self, original_prompt, refinement_plan, validation_result):
        """Create enhanced prompt based on comprehensive AI analysis"""
        refined_prompt = original_prompt
        
        # AI-driven prompt enhancements based on specific issues
        improvements = []
        
        # TIFA-based improvements
        tifa_score = validation_result.get('tifa_result', {}).get('score', 0.5)
        if tifa_score < 0.6:
            improvements.extend([
                "highly accurate representation",
                "exactly as described",
                "faithful to prompt"
            ])
        
        # Emotion-based improvements
        emotion_score = validation_result.get('emotion_analysis', {}).get('autism_appropriateness_score', 0.5)
        if emotion_score < 0.5:
            improvements.extend([
                "calm expression",
                "gentle demeanor",
                "friendly appearance",
                "peaceful emotion"
            ])
        
        # Background-based improvements
        bg_score = validation_result.get('background_analysis', {}).get('autism_background_score', 0.5)
        if bg_score < 0.5:
            improvements.extend([
                "simple background",
                "minimal distractions",
                "clean environment",
                "uncluttered scene"
            ])
        
        # Character clarity improvements
        face_count = validation_result.get('face_count', 0)
        if face_count == 0:
            improvements.extend([
                "clear visible character",
                "distinct facial features",
                "well-defined person"
            ])
        elif face_count > 2:
            improvements.extend([
                "single character focus",
                "one person only",
                "individual subject"
            ])
        
        # Add refinement plan suggestions
        if refinement_plan.get("refinement_prompt_additions"):
            improvements.extend(refinement_plan["refinement_prompt_additions"])
        
        # Combine improvements
        if improvements:
            unique_improvements = list(dict.fromkeys(improvements))  # Remove duplicates
            additions = ", ".join(unique_improvements[:5])  # Limit to top 5
            refined_prompt = f"{refined_prompt}, {additions}"
        
        # Always add quality terms for refinement
        if "high quality" not in refined_prompt.lower():
            refined_prompt = f"{refined_prompt}, high quality, detailed, clear"
        
        return refined_prompt
    
    def create_ai_refinement_negative(self, base_negative, refinement_plan, validation_result):
        """Create enhanced negative prompt based on comprehensive AI analysis"""
        base_negative = base_negative or "blurry, low quality, distorted, deformed"
        negative_additions = []
        
        # Add emotion-based negatives
        emotion_score = validation_result.get('emotion_analysis', {}).get('autism_appropriateness_score', 0.5)
        if emotion_score < 0.5:
            negative_additions.extend([
                "scary expression",
                "angry face",
                "intense emotions",
                "overwhelming feelings"
            ])
        
        # Add background-based negatives
        bg_score = validation_result.get('background_analysis', {}).get('autism_background_score', 0.5)
        if bg_score < 0.5:
            negative_additions.extend([
                "cluttered background",
                "busy scene",
                "many objects",
                "distracting elements"
            ])
        
        # Add character-based negatives
        face_count = validation_result.get('face_count', 0)
        if face_count == 0:
            negative_additions.extend([
                "hidden face",
                "no visible character",
                "obscured person"
            ])
        elif face_count > 2:
            negative_additions.extend([
                "multiple people",
                "crowd",
                "many faces",
                "group scene"
            ])
        
        # Add refinement plan negatives
        if refinement_plan.get("refinement_negative_additions"):
            negative_additions.extend(refinement_plan["refinement_negative_additions"])
        
        # Combine negatives
        if negative_additions:
            unique_negatives = list(dict.fromkeys(negative_additions))  # Remove duplicates
            additions = ", ".join(unique_negatives[:5])  # Limit to top 5
            refined_negative = f"{base_negative}, {additions}"
        else:
            refined_negative = base_negative
        
        return refined_negative
    
    def refine_image_with_ai_analysis(self, image, original_prompt, validation_result, base_negative=None):
        """Refine image using comprehensive AI analysis"""
        
        if not self.available:
            print("❌ AI refiner not available")
            return image
        
        # Generate refinement plan from validation
        refinement_plan = {
            "needs_refinement": False,
            "refinement_prompt_additions": [],
            "refinement_negative_additions": [],
            "refinement_strength": 0.3
        }
        
        # Determine if refinement is needed based on scores
        overall_score = validation_result.get('quality_score', 0.5)
        tifa_score = validation_result.get('tifa_result', {}).get('score', 0.5)
        emotion_score = validation_result.get('emotion_analysis', {}).get('autism_appropriateness_score', 0.5)
        bg_score = validation_result.get('background_analysis', {}).get('autism_background_score', 0.5)
        
        if overall_score < 0.65 or tifa_score < 0.6 or emotion_score < 0.5 or bg_score < 0.5:
            refinement_plan["needs_refinement"] = True
            
            # Determine refinement strength based on how poor the scores are
            if overall_score < 0.4:
                refinement_plan["refinement_strength"] = 0.6  # Strong refinement
            elif overall_score < 0.55:
                refinement_plan["refinement_strength"] = 0.4  # Medium refinement
            else:
                refinement_plan["refinement_strength"] = 0.3  # Light refinement
        
        if not refinement_plan["needs_refinement"]:
            print("✅ AI analysis: No img2img refinement needed")
            return image
        
        print(f"🔧 AI-driven img2img refinement (strength: {refinement_plan['refinement_strength']})")
        
        # Create AI-informed prompts
        refined_prompt = self.create_ai_refinement_prompt(original_prompt, refinement_plan, validation_result)
        refined_negative = self.create_ai_refinement_negative(base_negative, refinement_plan, validation_result)
        
        print(f"📝 AI-enhanced refinement prompt: {refined_prompt}")
        print(f"🚫 AI-enhanced refinement negative: {refined_negative}")
        
        refinement_settings = {
            'prompt': refined_prompt,
            'negative_prompt': refined_negative,
            'image': image,
            'strength': refinement_plan["refinement_strength"],
            'num_inference_steps': 25,
            'guidance_scale': 6.5,
            'generator': torch.Generator("cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())
        }
        
        try:
            print(f"🎨 Applying AI img2img refinement...")
            start_time = time.time()
            result = self.refiner_pipe(**refinement_settings)
            refinement_time = time.time() - start_time
            
            print(f"✅ AI img2img refinement completed in {refinement_time:.2f}s")
            return result.images[0]
            
        except Exception as e:
            print(f"❌ AI img2img refinement failed: {e}")
            return image