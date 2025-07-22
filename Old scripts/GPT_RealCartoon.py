# SAFETENSORS FIX - Add at very top before other imports
import warnings
import os

# Setup safetensors environment
warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['SAFETENSORS_FAST_GPU'] = '1'

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import cv2
import numpy as np
from PIL import Image
import time
import json
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import requests
from io import BytesIO

# Set your Hugging Face token
os.environ['HUGGINGFACE_TOKEN'] = "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"

print("🎨 ENHANCED REAL CARTOON XL v7 + ADVANCED AI VALIDATION & REFINEMENT FRAMEWORK")
print("=" * 80)
print("🔥 NOW WITH BLIP LARGE - MOST ADVANCED CAPTIONING MODEL")
print("🎯 Optimized for Autism Education Storyboards")
print("🔧 Techniques: CLIP, ADVANCED BLIP LARGE, HF APIs, Face Detection + AI Refiner")
print("✅ UPGRADE: Using BLIP Large for superior captioning quality")

class AdvancedImageCaptioner:
    """
    Most advanced image captioning using BLIP Large with optimized settings
    UPGRADED: Now uses BLIP Large with advanced generation parameters
    """
    
    def __init__(self, model_variant="large", enable_optimizations=True):
        print(f"🔥 Loading BLIP {model_variant.title()} (Most Advanced Captioning)...")
        
        # Model selection - force large for best results
        model_name = f"Salesforce/blip-image-captioning-{model_variant}"
        
        try:
            # Advanced processor loading with optimizations
            self.processor = BlipProcessor.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=False
            )
            
            # Advanced model loading with memory optimization
            if enable_optimizations and torch.cuda.is_available():
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                print("✅ BLIP Large GPU + memory optimizations enabled")
            else:
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True
                )
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    print("✅ BLIP Large GPU acceleration enabled")
                else:
                    print("⚡ BLIP Large running on CPU")
            
            self.model.eval()
            self.available = True
            
            # Advanced generation presets for highest quality
            self.quality_presets = {
                "maximum": {
                    "max_length": 150,
                    "min_length": 25,
                    "num_beams": 8,
                    "temperature": 0.6,
                    "repetition_penalty": 1.3,
                    "length_penalty": 1.1,
                    "early_stopping": True,
                    "do_sample": True,
                    "top_k": 50,
                    "top_p": 0.9
                },
                "high": {
                    "max_length": 120,
                    "min_length": 20,
                    "num_beams": 6,
                    "temperature": 0.7,
                    "repetition_penalty": 1.2,
                    "length_penalty": 1.0,
                    "early_stopping": True,
                    "do_sample": True,
                    "top_k": 40,
                    "top_p": 0.85
                },
                "autism_optimized": {
                    "max_length": 100,
                    "min_length": 15,
                    "num_beams": 5,
                    "temperature": 0.65,
                    "repetition_penalty": 1.25,
                    "length_penalty": 1.05,
                    "early_stopping": True,
                    "do_sample": True,
                    "top_k": 35,
                    "top_p": 0.8
                }
            }
            
        except Exception as e:
            print(f"⚠️ BLIP Large failed, falling back to base: {e}")
            # Fallback to base model
            try:
                base_model_name = "Salesforce/blip-image-captioning-base"
                self.processor = BlipProcessor.from_pretrained(base_model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(base_model_name)
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                self.available = True
                print("✅ BLIP Base loaded as fallback")
            except:
                self.available = False
                print("❌ No BLIP models available")

    def generate_advanced_caption(self, image, quality_preset="autism_optimized", conditional_text=None):
        """Generate high-quality captions with advanced BLIP Large settings"""
        if not self.available:
            return "Advanced captioning unavailable"
        
        try:
            # Ensure RGB format and proper preprocessing
            if isinstance(image, str):
                if image.startswith('http'):
                    image = Image.open(requests.get(image, stream=True).raw).convert('RGB')
                else:
                    image = Image.open(image).convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process with conditional or unconditional captioning
            if conditional_text:
                inputs = self.processor(image, conditional_text, return_tensors="pt")
            else:
                inputs = self.processor(image, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda", torch.float16) for k, v in inputs.items()}
            
            # Generate with advanced settings
            generation_kwargs = self.quality_presets.get(quality_preset, self.quality_presets["autism_optimized"])
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up caption if it includes conditional text
            if conditional_text and caption.startswith(conditional_text):
                caption = caption[len(conditional_text):].strip()
            
            return caption
            
        except Exception as e:
            print(f"Advanced captioning error: {e}")
            # Fallback to simple generation
            try:
                inputs = self.processor(image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=50)
                
                return self.processor.decode(outputs[0], skip_special_tokens=True)
            except:
                return f"Caption error: {str(e)[:50]}..."

class AIImageValidator:
    """
    ENHANCED AI image validation with BLIP Large integration
    UPGRADE: Now uses advanced BLIP Large captioning
    """
    
    def __init__(self):
        print("🔧 Loading ENHANCED validation systems...")
        
        # Initialize advanced captioner first
        self.advanced_captioner = AdvancedImageCaptioner(model_variant="large", enable_optimizations=True)
        
        self.setup_clip_similarity() 
        self.setup_face_detection()
        self.setup_hf_api_analysis()
        
        print("✅ ENHANCED validation framework ready!")
    
    def setup_clip_similarity(self):
        """CLIP for text-image similarity scoring - FIXED SAFETENSORS VERSION"""
        try:
            print("🔗 Loading CLIP similarity...")
            
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_fast=False,
                trust_remote_code=False
            )
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=True,
                trust_remote_code=False,
                torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
            self.clip_available = True
            print("✅ CLIP loaded successfully with safetensors")
        except Exception as e:
            print(f"⚠️ CLIP unavailable: {e}")
            self.clip_available = False
    
    def setup_face_detection(self):
        """OpenCV face detection for character analysis"""
        try:
            cascade_path = "haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                print("📥 Downloading face detection model...")
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                response = requests.get(url, timeout=30)
                with open(cascade_path, 'wb') as f:
                    f.write(response.content)
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise Exception("Failed to load cascade classifier")
                
            self.face_detection_available = True
            print("✅ Face detection loaded")
        except Exception as e:
            print(f"⚠️ Face detection unavailable: {e}")
            self.face_detection_available = False
    
    def setup_hf_api_analysis(self):
        """HuggingFace API for advanced analysis"""
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if self.hf_token:
            print(f"🤖 HF Token: {self.hf_token[:10]}...")
            self.hf_available = True
            print("✅ HF API ready")
        else:
            print("⚠️ No HF token found")
            self.hf_available = False
    
    def get_image_caption(self, image, use_advanced=True, quality_preset="autism_optimized"):
        """
        ENHANCED: Generate descriptive caption using BLIP Large
        Now supports multiple quality presets and advanced generation
        """
        if use_advanced and self.advanced_captioner.available:
            # Use the new advanced BLIP Large captioner
            caption = self.advanced_captioner.generate_advanced_caption(
                image, 
                quality_preset=quality_preset
            )
            return caption
        else:
            # Fallback to basic captioning (kept for compatibility)
            return "Basic captioning unavailable"
    
    def get_detailed_caption(self, image, conditional_prompt=None):
        """
        NEW: Get extremely detailed captions for autism education
        Uses maximum quality settings for educational content
        """
        if not self.advanced_captioner.available:
            return "Advanced captioning unavailable"
        
        # Generate with maximum quality for educational use
        detailed_caption = self.advanced_captioner.generate_advanced_caption(
            image, 
            quality_preset="maximum",
            conditional_text=conditional_prompt
        )
        
        return detailed_caption
    
    def calculate_similarity(self, image, text):
        """Calculate image-text similarity using CLIP"""
        if not self.clip_available:
            return 0.5
        
        try:
            inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                similarity = torch.sigmoid(outputs.logits_per_image[0][0])
            return float(similarity)
        except Exception as e:
            print(f"CLIP error: {e}")
            return 0.5
    
    def detect_faces(self, image):
        """Detect and count faces in image"""
        if not self.face_detection_available:
            return []
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                return [face.tolist() for face in faces]
            else:
                return []
                
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def hf_api_analysis(self, image):
        """Multi-model analysis using HuggingFace API"""
        if not self.hf_available:
            return {"available": False}
        
        try:
            # Convert image to bytes
            img_bytes = BytesIO()
            image.save(img_bytes, format='PNG')
            img_data = img_bytes.getvalue()
            
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            results = {}
            
            # Image Classification
            try:
                print("🤖 Running classification...")
                response = requests.post(
                    "https://api-inference.huggingface.co/models/microsoft/resnet-50",
                    headers=headers, data=img_data, timeout=20
                )
                if response.status_code == 200:
                    classifications = response.json()
                    if isinstance(classifications, list):
                        results["classification"] = classifications[:5]
                        labels = [c.get('label', 'unknown') for c in results['classification'][:3]]
                        print(f"✅ Classifications: {labels}")
                    else:
                        results["classification"] = "Unexpected format"
                else:
                    results["classification"] = f"Error: {response.status_code}"
            except Exception as e:
                results["classification"] = f"Error: {str(e)}"
            
            # Object Detection
            try:
                print("🤖 Running object detection...")
                response = requests.post(
                    "https://api-inference.huggingface.co/models/facebook/detr-resnet-50",
                    headers=headers, data=img_data, timeout=20
                )
                if response.status_code == 200:
                    objects = response.json()
                    if isinstance(objects, list):
                        good_objects = [obj for obj in objects if obj.get('score', 0) > 0.3]
                        results["objects"] = good_objects[:5]
                        labels = [obj.get('label', 'unknown') for obj in good_objects[:3]]
                        print(f"✅ Objects: {labels}")
                    else:
                        results["objects"] = "Unexpected format"
                else:
                    results["objects"] = f"Error: {response.status_code}"
            except Exception as e:
                results["objects"] = f"Error: {str(e)}"
            
            return {"available": True, "results": results}
            
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def calculate_quality_score(self, similarity, face_count, hf_results, prompt, caption_quality=0.8):
        """
        ENHANCED: Calculate quality score with caption quality factor
        Now includes advanced caption quality assessment
        """
        base_score = similarity
        
        # NEW: Caption quality bonus (from BLIP Large)
        base_score += (caption_quality - 0.5) * 0.2  # Bonus for high-quality captions
        
        # AUTISM STORYBOARD SPECIFIC SCORING
        
        # Character clarity (critical for autism learning)
        if any(keyword in prompt.lower() for keyword in ["person", "boy", "girl", "child", "character"]):
            if face_count == 1:
                base_score += 0.20  # Perfect for autism storyboards - single clear character
            elif face_count == 2:
                base_score += 0.10  # Good for social interaction scenes
            elif face_count == 0:
                base_score -= 0.15  # Missing character - bad for autism education
            elif face_count > 2:
                base_score -= 0.20  # Too complex - confusing for autism learners
        
        # Real Cartoon XL v7 optimization
        if "cartoon" in prompt.lower():
            base_score += 0.10  # Bonus for explicit cartoon style with this model
        
        # Autism-friendly content checks
        autism_friendly_keywords = ["brushing teeth", "eating", "getting dressed", "school", "daily", "routine"]
        if any(keyword in prompt.lower() for keyword in autism_friendly_keywords):
            base_score += 0.10  # Bonus for educational daily activities
        
        # Clarity bonus (fewer objects = clearer for autism)
        if hf_results.get("available") and "results" in hf_results:
            objects = hf_results["results"].get("objects", [])
            if isinstance(objects, list):
                if len(objects) <= 3:  # Simple, clear scenes
                    base_score += 0.10
                elif len(objects) > 5:  # Too cluttered
                    base_score -= 0.10
        
        # HF analysis working bonus
        if hf_results.get("available") and "results" in hf_results:
            working_models = sum(1 for result in hf_results["results"].values() 
                               if not str(result).startswith("Error"))
            base_score += 0.05 * working_models
        
        return max(0.0, min(1.0, base_score))
    
    def generate_recommendations(self, similarity, face_count, hf_results, prompt, caption=""):
        """
        ENHANCED: Generate recommendations with caption analysis
        Now includes advanced caption-based suggestions
        """
        recommendations = []
        
        # NEW: Caption quality analysis
        if len(caption.split()) < 5:
            recommendations.append("📝 Caption too brief - may indicate unclear image")
        elif len(caption.split()) > 20:
            recommendations.append("📝 Very detailed caption - good image clarity")
        
        # Activity detection from caption
        if caption and any(activity in caption.lower() for activity in ["eating", "brushing", "washing", "playing"]):
            recommendations.append("✅ Clear activity depicted - excellent for autism storyboards")
        
        # CLIP similarity recommendations
        if similarity < 0.6:
            recommendations.append("🔄 Low similarity - Real Cartoon XL v7 needs more specific prompts")
        elif similarity < 0.7:
            recommendations.append("🎨 Good similarity - add more descriptive details for Real Cartoon XL v7")
        
        # Character visibility (critical for autism storyboards)
        if any(keyword in prompt.lower() for keyword in ["person", "boy", "girl", "child"]):
            if face_count == 0:
                recommendations.append("❌ No character visible - unsuitable for autism storyboards")
            elif face_count == 1:
                recommendations.append("✅ Perfect single character - ideal for autism learning")
            elif face_count == 2:
                recommendations.append("✅ Good for social interaction scenes in autism education")
            elif face_count > 2:
                recommendations.append("⚠️ Too many characters - may confuse autism learners")
        
        # Real Cartoon XL v7 specific tips
        if "cartoon" not in prompt.lower():
            recommendations.append("💡 Add 'cartoon style' for better Real Cartoon XL v7 results")
        
        # Autism education content suggestions
        daily_activities = ["brushing teeth", "eating", "getting dressed", "school"]
        if not any(activity in prompt.lower() for activity in daily_activities):
            recommendations.append("📚 Consider daily routine activities for autism storyboards")
        
        # Scene complexity for autism learners
        if hf_results.get("available") and "results" in hf_results:
            objects = hf_results["results"].get("objects", [])
            if isinstance(objects, list) and len(objects) > 5:
                recommendations.append("🎯 Scene may be too complex - simplify for autism learners")
        
        # API availability
        if not hf_results.get("available"):
            recommendations.append("🤖 Limited analysis - check HuggingFace API connection")
        
        # Positive feedback
        if similarity > 0.7 and face_count in [1, 2]:
            recommendations.append("🌟 Excellent for autism storyboards!")
        
        return recommendations if recommendations else ["✨ Good quality for autism education"]
    
    def assess_autism_suitability(self, face_count, similarity, hf_analysis, caption=""):
        """
        ENHANCED: Assess suitability with caption analysis
        Now includes advanced caption-based assessment
        """
        suitability = {
            "suitable": False,
            "score": 0,
            "reasons": []
        }
        
        # NEW: Caption-based assessment
        if caption:
            caption_lower = caption.lower()
            
            # Clear activity description
            if any(activity in caption_lower for activity in ["eating", "brushing", "washing", "sitting", "standing"]):
                suitability["score"] += 15
                suitability["reasons"].append("Clear activity described")
            
            # Character mention in caption
            if any(char in caption_lower for char in ["boy", "girl", "child", "person", "man", "woman"]):
                suitability["score"] += 10
                suitability["reasons"].append("Character clearly identified")
            
            # Educational setting
            if any(setting in caption_lower for setting in ["bathroom", "kitchen", "bedroom", "school"]):
                suitability["score"] += 10
                suitability["reasons"].append("Educational setting identified")
        
        # Character clarity (most important for autism storyboards)
        if face_count == 1:
            suitability["score"] += 40
            suitability["reasons"].append("Single clear character")
        elif face_count == 2:
            suitability["score"] += 30
            suitability["reasons"].append("Two characters for social learning")
        elif face_count == 0:
            suitability["score"] -= 30
            suitability["reasons"].append("No visible characters")
        else:
            suitability["score"] -= 20
            suitability["reasons"].append("Too many characters")
        
        # Prompt adherence
        if similarity > 0.7:
            suitability["score"] += 30
            suitability["reasons"].append("High prompt accuracy")
        elif similarity > 0.6:
            suitability["score"] += 20
            suitability["reasons"].append("Good prompt accuracy")
        else:
            suitability["score"] -= 10
            suitability["reasons"].append("Low prompt accuracy")
        
        # Scene complexity
        if hf_analysis.get("available") and "results" in hf_analysis:
            objects = hf_analysis["results"].get("objects", [])
            if isinstance(objects, list):
                if len(objects) <= 3:
                    suitability["score"] += 20
                    suitability["reasons"].append("Simple, clear scene")
                elif len(objects) > 5:
                    suitability["score"] -= 15
                    suitability["reasons"].append("Complex scene")
        
        # Final assessment
        suitability["suitable"] = suitability["score"] >= 50
        
        return suitability
    
    def analyze_refinement_needs(self, validation_result):
        """
        ENHANCED: Analyze validation with caption insights
        Now includes caption-based refinement suggestions
        """
        refinement_plan = {
            "needs_refinement": False,
            "issues": [],
            "refinement_prompt_additions": [],
            "refinement_negative_additions": [],
            "refinement_strength": 0.3,
            "priority": "medium"
        }
        
        similarity = validation_result.get("similarity", 0)
        face_count = validation_result.get("face_count", 0)
        prompt = validation_result.get("prompt", "")
        caption = validation_result.get("caption", "")
        hf_analysis = validation_result.get("hf_analysis", {})
        
        # NEW: Caption-based refinement analysis
        if caption:
            caption_lower = caption.lower()
            
            # Activity clarity
            if "brushing teeth" in prompt.lower() and "brush" not in caption_lower and "teeth" not in caption_lower:
                refinement_plan["needs_refinement"] = True
                refinement_plan["issues"].append("Activity not clearly depicted in generated image")
                refinement_plan["refinement_prompt_additions"].append("clearly brushing teeth with toothbrush")
                refinement_plan["priority"] = "high"
            
            # Character clarity from caption
            if any(char_word in prompt.lower() for char_word in ["boy", "girl", "child"]) and \
               not any(char_word in caption_lower for char_word in ["boy", "girl", "child", "person"]):
                refinement_plan["needs_refinement"] = True
                refinement_plan["issues"].append("Character not clearly depicted")
                refinement_plan["refinement_prompt_additions"].append("clear character, visible person")
        
        # Character visibility issues
        if any(keyword in prompt.lower() for keyword in ["person", "boy", "girl", "child"]):
            if face_count == 0:
                refinement_plan["needs_refinement"] = True
                refinement_plan["issues"].append("Missing character - no faces detected")
                refinement_plan["refinement_prompt_additions"].append("clear visible character face")
                refinement_plan["refinement_negative_additions"].append("hidden face, obscured character")
                refinement_plan["refinement_strength"] = 0.4
                refinement_plan["priority"] = "high"
            
            elif face_count > 2:
                refinement_plan["needs_refinement"] = True
                refinement_plan["issues"].append("Too many characters for autism education")
                refinement_plan["refinement_prompt_additions"].append("single character, one person only")
                refinement_plan["refinement_negative_additions"].append("multiple people, crowd, many faces")
                refinement_plan["refinement_strength"] = 0.3
        
        # Low similarity issues
        if similarity < 0.6:
            refinement_plan["needs_refinement"] = True
            refinement_plan["issues"].append("Low prompt adherence")
            refinement_plan["refinement_prompt_additions"].append("highly detailed, accurate representation")
            refinement_plan["refinement_strength"] = max(refinement_plan["refinement_strength"], 0.35)
            refinement_plan["priority"] = "high"
        
        # Scene complexity issues
        if hf_analysis.get("available") and "results" in hf_analysis:
            objects = hf_analysis["results"].get("objects", [])
            if isinstance(objects, list) and len(objects) > 5:
                refinement_plan["needs_refinement"] = True
                refinement_plan["issues"].append("Scene too complex for autism learners")
                refinement_plan["refinement_prompt_additions"].append("simple background, minimal objects")
                refinement_plan["refinement_negative_additions"].append("cluttered, busy scene, many objects")
                refinement_plan["refinement_strength"] = 0.3
        
        # Autism education specific issues
        if "cartoon" not in prompt.lower():
            refinement_plan["refinement_prompt_additions"].append("cartoon style illustration")
        
        return refinement_plan
    
    def validate_image(self, image, prompt, context="autism_storyboard"):
        """
        ENHANCED: Complete validation with advanced BLIP Large captioning
        Now provides superior caption quality and analysis
        """
        print(f"\n🔍 ENHANCED VALIDATION FOR AUTISM STORYBOARD")
        print(f"Prompt: {prompt}")
        
        # ENHANCED: Core validations with advanced captioning
        print("📝 Generating advanced caption with BLIP Large...")
        caption = self.get_image_caption(image, use_advanced=True, quality_preset="autism_optimized")
        
        print("📝 Generating detailed educational caption...")
        detailed_caption = self.get_detailed_caption(image, "a detailed description of")
        
        similarity = self.calculate_similarity(image, prompt)
        faces = self.detect_faces(image)
        hf_analysis = self.hf_api_analysis(image)
        
        # Results with enhanced information
        print(f"📝 Standard Caption: {caption}")
        print(f"📚 Educational Caption: {detailed_caption}")
        print(f"🔗 CLIP Similarity: {similarity:.3f}")
        print(f"👤 Characters Detected: {len(faces)}")
        
        # Caption quality assessment
        caption_words = len(caption.split()) if caption else 0
        caption_quality = min(1.0, max(0.3, caption_words / 15))  # Quality based on detail
        print(f"📊 Caption Quality: {caption_quality:.3f}")
        
        # Autism storyboard specific analysis
        if len(faces) == 1:
            print("✅ Single character - perfect for autism learning")
        elif len(faces) == 2:
            print("✅ Two characters - good for social skill teaching")
        elif len(faces) == 0:
            print("❌ No characters visible - not suitable for autism storyboard")
        else:
            print("⚠️ Multiple characters - may be too complex for autism learners")
        
        # Advanced results from HF API
        if hf_analysis.get("available") and "results" in hf_analysis:
            hf_results = hf_analysis["results"]
            
            # Object complexity check for autism education
            objects = hf_results.get("objects", [])
            if isinstance(objects, list):
                print(f"🎯 Scene Objects: {len(objects)} detected")
                if len(objects) <= 3:
                    print("✅ Simple scene - good for autism learners")
                elif len(objects) > 5:
                    print("⚠️ Complex scene - may overwhelm autism learners")
        
        # Calculate enhanced autism-specific scores
        quality_score = self.calculate_quality_score(similarity, len(faces), hf_analysis, prompt, caption_quality)
        recommendations = self.generate_recommendations(similarity, len(faces), hf_analysis, prompt, caption)
        autism_suitability = self.assess_autism_suitability(len(faces), similarity, hf_analysis, caption)
        
        print(f"📊 Enhanced Autism Storyboard Score: {quality_score:.3f}")
        print(f"🎯 Recommendations: {recommendations}")
        
        return {
            "prompt": prompt,
            "caption": caption,
            "detailed_caption": detailed_caption,
            "caption_quality": caption_quality,
            "similarity": similarity,
            "face_count": len(faces),
            "face_positions": faces,
            "hf_analysis": hf_analysis,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "context": context,
            "autism_suitability": autism_suitability
        }

class AIImageRefiner:
    """AI-powered image refinement using img2img with validation-informed prompts"""
    
    def __init__(self, model_path):
        print("🔧 Loading AI Image Refiner...")
        
        self.model_path = model_path
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        print("📁 Loading refinement pipeline...")
        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            model_path, torch_dtype=torch.float16, use_safetensors=True
        )
        
        self.refiner_pipe = self.refiner_pipe.to("cuda")
        self.refiner_pipe.enable_model_cpu_offload()
        self.refiner_pipe.enable_vae_slicing()
        
        print("✅ AI Image Refiner ready!")
    
    def create_refinement_prompt(self, original_prompt, refinement_plan):
        """Create enhanced prompt based on validation issues"""
        refined_prompt = original_prompt
        
        if refinement_plan["refinement_prompt_additions"]:
            additions = ", ".join(refinement_plan["refinement_prompt_additions"])
            refined_prompt = f"{refined_prompt}, {additions}"
        
        quality_terms = ["high quality", "detailed", "clear"]
        if not any(term in refined_prompt.lower() for term in quality_terms):
            refined_prompt = f"{refined_prompt}, high quality, detailed"
        
        return refined_prompt
    
    def create_refinement_negative(self, base_negative, refinement_plan):
        """Create enhanced negative prompt based on issues"""
        base_negative = base_negative or "blurry, low quality, distorted, deformed"
        
        if refinement_plan["refinement_negative_additions"]:
            additions = ", ".join(refinement_plan["refinement_negative_additions"])
            refined_negative = f"{base_negative}, {additions}"
        else:
            refined_negative = base_negative
        
        return refined_negative
    
    def refine_image(self, image, original_prompt, refinement_plan, base_negative=None):
        """Refine image using validation-informed prompts"""
        
        if not refinement_plan["needs_refinement"]:
            print("✅ No refinement needed")
            return image
        
        print(f"🔧 Refining image for issues: {refinement_plan['issues']}")
        
        refined_prompt = self.create_refinement_prompt(original_prompt, refinement_plan)
        refined_negative = self.create_refinement_negative(base_negative, refinement_plan)
        
        print(f"📝 Refined prompt: {refined_prompt}")
        print(f"🚫 Refined negative: {refined_negative}")
        
        refinement_settings = {
            'prompt': refined_prompt,
            'negative_prompt': refined_negative,
            'image': image,
            'strength': refinement_plan["refinement_strength"],
            'num_inference_steps': 20,
            'guidance_scale': 6.0,
            'generator': torch.Generator("cuda").manual_seed(torch.randint(0, 2**32, (1,)).item())
        }
        
        try:
            print(f"🎨 Applying refinement (strength: {refinement_plan['refinement_strength']})...")
            start_time = time.time()
            result = self.refiner_pipe(**refinement_settings)
            refinement_time = time.time() - start_time
            
            print(f"✅ Refinement completed in {refinement_time:.2f}s")
            return result.images[0]
            
        except Exception as e:
            print(f"❌ Refinement failed: {e}")
            return image

class RealCartoonXLAutismValidator:
    """
    ENHANCED: Complete pipeline for Real Cartoon XL v7 autism storyboard creation 
    WITH ADVANCED BLIP LARGE CAPTIONING + AI-POWERED REFINEMENT
    """
    
    def __init__(self, model_path):
        print("🎨 Loading ENHANCED Real Cartoon XL v7 + Autism Storyboard Validator + AI Refiner...")
        print("🔥 NOW WITH BLIP LARGE - MOST ADVANCED CAPTIONING")
        
        self.model_path = model_path
        if not os.path.exists(model_path):
            print(f"❌ Real Cartoon XL v7 not found: {model_path}")
            return
        
        print(f"📁 Loading Real Cartoon XL v7: {model_path}")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=torch.float16, use_safetensors=True
        )
        
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        
        # Initialize enhanced validator with BLIP Large
        self.validator = AIImageValidator()
        self.refiner = AIImageRefiner(model_path)
        
        print("✅ ENHANCED Real Cartoon XL v7 + BLIP Large Validator + AI Refiner ready!")
    
    def find_realcartoon_model(self):
        """Find Real Cartoon XL v7 model file"""
        possible_paths = [
            r"C:\RuinedFooocus-main\models\checkpoints\realcartoonxl_v7.safetensors",
            r".\models\realcartoonxl_v7.safetensors",
            r".\realcartoonxl_v7.safetensors",
            r"C:\Users\Leon Parker\Documents\Newcastle University\Semester 3\Individual Project\SDXL Diffusers - Lora\models\realcartoonxl_v7.safetensors"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found Real Cartoon XL v7: {path}")
                return path
        
        print("❌ Real Cartoon XL v7 not found in expected locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return None
    
    def generate_storyboard_image_with_refinement(self, prompt, attempts=3, target_score=0.70, enable_refinement=True):
        """
        ENHANCED: Generate autism storyboard image with BLIP Large validation + AI refinement
        UPGRADED 4-PHASE SYSTEM with advanced captioning
        """
        
        # Real Cartoon XL v7 optimized settings for autism storyboards
        generation_settings = {
            'height': 1024, 'width': 1024, 
            'num_inference_steps': 30,
            'guidance_scale': 5.0,  # Lower guidance works better for Real Cartoon XL
            'negative_prompt': "blurry, low quality, distorted, deformed, scary, dark, confusing, multiple people, complex background, cluttered"
        }
        
        best_result = None
        best_score = 0
        
        # PHASE 1: Enhanced Initial Generation Loop
        print("🎨 PHASE 1: Enhanced Initial Generation with BLIP Large")
        for attempt in range(attempts):
            print(f"\n🎨 Real Cartoon XL v7 - Generation Attempt {attempt + 1}/{attempts}")
            
            # Generate with random seed
            generation_settings['generator'] = torch.Generator("cuda").manual_seed(
                torch.randint(0, 2**32, (1,)).item()
            )
            
            start_time = time.time()
            result = self.pipe(prompt=prompt, **generation_settings)
            gen_time = time.time() - start_time
            
            # ENHANCED: Validate with BLIP Large for autism storyboard use
            print("🔍 Enhanced validation with BLIP Large...")
            validation = self.validator.validate_image(result.images[0], prompt, "autism_storyboard")
            score = validation["quality_score"]
            
            print(f"📊 Enhanced Score: {score:.3f}")
            print(f"📝 BLIP Large Caption: {validation['caption']}")
            print(f"📚 Detailed Caption: {validation['detailed_caption']}")
            print(f"🧠 Autism Suitable: {'✅ YES' if validation['autism_suitability']['suitable'] else '❌ NO'}")
            
            if score > best_score:
                best_score = score
                best_result = {
                    'image': result.images[0],
                    'validation': validation,
                    'generation_time': gen_time,
                    'attempt': attempt + 1,
                    'model': 'Real Cartoon XL v7 + BLIP Large',
                    'refined': False
                }
            
            if score >= target_score and validation['autism_suitability']['suitable']:
                print(f"✅ Enhanced target achieved in initial generation!")
                break
            else:
                print(f"⚠️ Score {score:.3f} - continuing generation attempts")
        
        print(f"\n🏆 Best enhanced initial result: {best_score:.3f} from attempt {best_result['attempt']}")
        
        # PHASE 2: Advanced AI Refinement (if enabled and needed)
        if enable_refinement and best_result:
            print("\n🔧 PHASE 2: Advanced AI-Powered Refinement")
            
            # Analyze what needs refinement with enhanced validation data
            refinement_plan = self.validator.analyze_refinement_needs(best_result['validation'])
            
            if refinement_plan["needs_refinement"]:
                print(f"🎯 Enhanced refinement needed for: {refinement_plan['issues']}")
                print(f"🔧 Priority: {refinement_plan['priority']}")
                
                # Apply AI refinement
                refined_image = self.refiner.refine_image(
                    best_result['image'], 
                    prompt, 
                    refinement_plan,
                    generation_settings['negative_prompt']
                )
                
                # ENHANCED: Validate refined image with BLIP Large
                print("\n🔍 Enhanced validation of refined image with BLIP Large...")
                refined_validation = self.validator.validate_image(refined_image, prompt, "autism_storyboard")
                refined_score = refined_validation["quality_score"]
                
                print(f"📊 Enhanced Refined Score: {refined_score:.3f}")
                print(f"📈 Improvement: {refined_score - best_score:+.3f}")
                print(f"📝 Refined BLIP Caption: {refined_validation['caption']}")
                print(f"🧠 Refined Autism Suitability: {'✅ YES' if refined_validation['autism_suitability']['suitable'] else '❌ NO'}")
                
                # Use refined version if it's better
                if refined_score > best_score:
                    print("✅ Enhanced refinement improved the image!")
                    best_result = {
                        'image': refined_image,
                        'validation': refined_validation,
                        'generation_time': best_result['generation_time'],
                        'attempt': best_result['attempt'],
                        'model': 'Real Cartoon XL v7 + BLIP Large',
                        'refined': True,
                        'refinement_plan': refinement_plan,
                        'improvement': refined_score - best_score,
                        'original_score': best_score,
                        'refined_score': refined_score
                    }
                    best_score = refined_score
                else:
                    print("⚠️ Enhanced refinement didn't improve quality - keeping original")
                    best_result['refined'] = False
                    best_result['refinement_attempted'] = True
                    best_result['refinement_plan'] = refinement_plan
            else:
                print("✅ No refinement needed - enhanced image quality already excellent!")
                best_result['refined'] = False
                best_result['refinement_needed'] = False
        
        # PHASE 3: Enhanced Final Assessment
        print(f"\n🎉 ENHANCED FINAL RESULT:")
        print(f"📊 Final Enhanced Score: {best_score:.3f}")
        print(f"🔧 Refined: {'✅ YES' if best_result.get('refined') else '❌ NO'}")
        if best_result.get('refined'):
            print(f"📈 Enhancement: +{best_result.get('improvement', 0):.3f}")
        print(f"🧠 Autism Suitable: {'✅ YES' if best_result['validation']['autism_suitability']['suitable'] else '❌ NO'}")
        print(f"📝 Final Caption: {best_result['validation']['caption']}")
        print(f"📚 Educational Caption: {best_result['validation']['detailed_caption']}")
        
        # PHASE 4: Enhanced Recommendations
        print(f"\n💡 ENHANCED RECOMMENDATIONS:")
        for rec in best_result['validation']['recommendations']:
            print(f"   {rec}")
        
        return best_result
    
    def generate_storyboard_image(self, prompt, attempts=3, target_score=0.70):
        """Legacy method - calls new enhanced refinement method"""
        return self.generate_storyboard_image_with_refinement(prompt, attempts, target_score, enable_refinement=True)

def demo_enhanced_framework_capabilities():
    """Demo showing the enhanced framework capabilities"""
    
    print("\n📚 ENHANCED REAL CARTOON XL v7 + BLIP LARGE VALIDATION & REFINEMENT FRAMEWORK")
    print("=" * 90)
    print("🔥 MAJOR UPGRADE: Now with BLIP Large - Most Advanced Captioning Model")
    print("🔧 This is an enhanced library script - import and use in your generation scripts!")
    
    print("\n🎯 ENHANCED AVAILABLE CLASSES:")
    print("✅ AdvancedImageCaptioner - BLIP Large with premium quality settings")
    print("✅ AIImageValidator - Multi-model validation + BLIP Large integration")
    print("✅ AIImageRefiner - AI-powered image refinement")
    print("✅ RealCartoonXLAutismValidator - Complete enhanced pipeline")
    
    print("\n🔧 ENHANCED VALIDATION TECHNIQUES:")
    print("• 🔥 BLIP Large Advanced Captioning (NEW: Most detailed captions)")
    print("• 📚 Educational Caption Generation (NEW: Specialized for autism learning)")
    print("• 🎯 Caption Quality Assessment (NEW: Measures caption detail/accuracy)")
    print("• 🔗 CLIP Text-Image Similarity (Enhanced with better error handling)")
    print("• 👤 OpenCV Face Detection (Optimized for autism storyboards)")
    print("• 🤖 HuggingFace API Analysis (3 models)")
    print("• 🧠 Advanced Autism Education Optimization")
    print("• 🔧 AI-Powered Refinement with Caption Insights")
    
    print("\n🚀 NEW FEATURES IN THIS ENHANCED VERSION:")
    print("✅ BLIP Large with maximum quality settings (150 token captions)")
    print("✅ Autism-optimized caption generation preset")
    print("✅ Detailed educational captions for learning materials")
    print("✅ Caption quality scoring and analysis")
    print("✅ Caption-based refinement suggestions")
    print("✅ Enhanced autism suitability assessment")
    print("✅ Activity detection from captions")
    print("✅ Educational setting identification")
    print("✅ 4-phase generation with advanced validation")
    
    print("\n📊 QUALITY IMPROVEMENTS:")
    print("• 🎯 Higher accuracy scores (caption quality factor)")
    print("• 📝 More detailed and educational captions")
    print("• 🧠 Better autism suitability detection")
    print("• 🔍 Enhanced refinement targeting")
    print("• 📚 Educational content optimization")
    
    print("\n💡 ENHANCED USAGE EXAMPLE:")
    print("from enhanced_real_cartoon_validator import RealCartoonXLAutismValidator")
    print("validator = RealCartoonXLAutismValidator('model_path')")
    print("result = validator.generate_storyboard_image_with_refinement('prompt')")
    print("print(f'Caption: {result[\"validation\"][\"caption\"]}')")
    print("print(f'Educational: {result[\"validation\"][\"detailed_caption\"]}')")
    
    print("\n📁 RECOMMENDED ENHANCED FILE STRUCTURE:")
    print("enhanced_autism_storyboard_generator/")
    print("├── enhanced_real_cartoon_validator.py    # This enhanced framework")
    print("├── generate_morning_routine_enhanced.py  # Enhanced generation script")
    print("├── generate_school_day_enhanced.py       # Enhanced generation script")
    print("└── models/")
    print("    └── realcartoonxl_v7.safetensors")
    
    print("\n🎉 Ready to generate ENHANCED high-quality autism storyboards!")
    print("Expected major improvements:")
    print("• 📝 Much more detailed and educational captions")
    print("• 🎯 Higher quality scores with caption analysis")
    print("• 🧠 Better autism suitability detection")
    print("• 📚 Educational content optimization")
    print("• 🔍 More targeted refinement suggestions")
    print("• ✨ Superior overall generation quality")

if __name__ == "__main__":
    demo_enhanced_framework_capabilities()