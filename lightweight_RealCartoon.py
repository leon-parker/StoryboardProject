# STREAMLINED REAL CARTOON XL v7 + TIFA + CLIP CONSISTENCY
# Core Features: TIFA scoring, CLIP embeddings, iterative prompt improvement

import warnings
import os
import torch
from diffusers import StableDiffusionXLPipeline
import numpy as np
from PIL import Image
import time
import json
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, 
    CLIPProcessor, CLIPModel
)
from sentence_transformers import SentenceTransformer
import spacy

# Setup environment
warnings.filterwarnings("ignore")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['SAFETENSORS_FAST_GPU'] = '1'

print("🎨 STREAMLINED REAL CARTOON XL v7 + TIFA + CLIP CONSISTENCY")
print("=" * 70)
print("🎯 Core Features: TIFA scoring, CLIP embeddings, iterative improvement")

class TIFAMetric:
    """Text-Image Faithfulness Assessment using AI models"""
    
    def __init__(self):
        print("🎯 Loading TIFA metric...")
        
        try:
            # CLIP for visual-semantic alignment
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            
            # NLP model for semantic parsing
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_available = True
                print("✅ SpaCy NLP loaded")
            except:
                print("⚠️ SpaCy not available - using basic parsing")
                self.spacy_available = False
            
            # Sentence transformer for semantic embeddings
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
                
            self.available = True
            print("✅ TIFA metric loaded")
            
        except Exception as e:
            print(f"⚠️ TIFA unavailable: {e}")
            self.available = False
    
    def extract_semantic_components(self, text):
        """Extract semantic components using AI models"""
        components = {
            'entities': [],
            'actions': [],
            'attributes': [],
            'concepts': []
        }
        
        if self.spacy_available:
            doc = self.nlp(text)
            
            # Extract entities
            for ent in doc.ents:
                components['entities'].append({
                    'text': ent.text.lower(),
                    'label': ent.label_
                })
            
            # Extract actions/verbs
            for token in doc:
                if token.pos_ == "VERB" and not token.is_stop:
                    components['actions'].append({
                        'text': token.lemma_.lower()
                    })
                
                # Extract attributes
                elif token.pos_ == "ADJ" and not token.is_stop:
                    components['attributes'].append({
                        'text': token.lemma_.lower()
                    })
            
            # Extract noun concepts
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:
                    components['concepts'].append({
                        'text': chunk.text.lower()
                    })
        else:
            # Basic parsing fallback
            words = text.lower().split()
            for word in words:
                components['concepts'].append({'text': word})
        
        return components
    
    def calculate_semantic_overlap(self, prompt_components, caption_components):
        """Calculate semantic overlap using embeddings"""
        overlap_scores = {}
        
        for component_type in prompt_components.keys():
            prompt_items = prompt_components[component_type]
            caption_items = caption_components[component_type]
            
            if not prompt_items:
                continue
            
            # Convert to text for embedding
            prompt_texts = [item['text'] if isinstance(item, dict) else str(item) for item in prompt_items]
            caption_texts = [item['text'] if isinstance(item, dict) else str(item) for item in caption_items]
            
            if not caption_texts:
                overlap_scores[component_type] = 0.0
                continue
            
            # Use AI embeddings to calculate semantic similarity
            try:
                prompt_embeddings = self.sentence_encoder.encode(prompt_texts)
                caption_embeddings = self.sentence_encoder.encode(caption_texts)
                
                # Calculate maximum semantic similarity
                max_similarities = []
                for p_emb in prompt_embeddings:
                    similarities = [np.dot(p_emb, c_emb) / (np.linalg.norm(p_emb) * np.linalg.norm(c_emb)) 
                                  for c_emb in caption_embeddings]
                    max_similarities.append(max(similarities) if similarities else 0.0)
                
                overlap_scores[component_type] = np.mean(max_similarities)
                
            except Exception as e:
                print(f"Embedding calculation error for {component_type}: {e}")
                overlap_scores[component_type] = 0.0
        
        return overlap_scores
    
    def calculate_tifa_score(self, image, prompt, generated_caption):
        """Calculate TIFA score"""
        if not self.available:
            return {"score": 0.5, "details": "TIFA unavailable"}
        
        try:
            # Extract semantic components
            prompt_components = self.extract_semantic_components(prompt)
            caption_components = self.extract_semantic_components(generated_caption)
            
            # Calculate semantic overlap
            semantic_overlaps = self.calculate_semantic_overlap(prompt_components, caption_components)
            
            # CLIP semantic similarity
            inputs = self.clip_processor(
                text=[prompt, generated_caption], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
                img_prompt_sim = torch.cosine_similarity(
                    outputs.image_embeds, 
                    outputs.text_embeds[0:1], 
                    dim=1
                ).item()
                
                prompt_caption_sim = torch.cosine_similarity(
                    outputs.text_embeds[0:1], 
                    outputs.text_embeds[1:2], 
                    dim=1
                ).item()
            
            # Final score calculation
            clip_score = img_prompt_sim
            semantic_score = np.mean(list(semantic_overlaps.values())) if semantic_overlaps else 0.5
            
            # Weighted combination
            tifa_score = (clip_score * 0.8) + (semantic_score * 0.2)
            
            return {
                "score": float(tifa_score),
                "details": {
                    "clip_score": float(clip_score),
                    "semantic_score": float(semantic_score),
                    "component_overlaps": semantic_overlaps,
                    "image_prompt_similarity": float(img_prompt_sim),
                    "prompt_caption_similarity": float(prompt_caption_sim)
                }
            }
            
        except Exception as e:
            print(f"TIFA calculation error: {e}")
            return {"score": 0.5, "details": f"Error: {e}"}

class CLIPMemoryManager:
    """Manages CLIP embeddings for consistency across prompts and images"""
    
    def __init__(self):
        print("🧠 Loading CLIP memory manager...")
        
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
            
            self.memory_bank = {}  # Store embeddings for consistency
            self.selected_images_history = []  # Track selected images across prompts
            self.style_memory = None  # Store style embedding from first image
            self.character_memory = None  # Store character embedding
            self.available = True
            print("✅ CLIP memory manager loaded")
            
        except Exception as e:
            print(f"⚠️ CLIP memory manager unavailable: {e}")
            self.available = False
    
    def get_text_embedding(self, text):
        """Get CLIP embedding for text"""
        if not self.available:
            return None
        
        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                return text_features.cpu().numpy()
        except Exception as e:
            print(f"Text embedding error: {e}")
            return None
    
    def get_image_embedding(self, image):
        """Get CLIP embedding for image"""
        if not self.available:
            return None
        
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                return image_features.cpu().numpy()
        except Exception as e:
            print(f"Image embedding error: {e}")
            return None
    
    def store_embedding(self, key, embedding, embedding_type="text", metadata=None):
        """Store embedding in memory bank with metadata"""
        self.memory_bank[key] = {
            "embedding": embedding,
            "type": embedding_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
    
    def store_selected_image(self, image, prompt, tifa_score, prompt_index=None):
        """Store selected image embedding for cross-prompt consistency"""
        if not self.available:
            return
        
        image_embedding = self.get_image_embedding(image)
        if image_embedding is None:
            return
        
        # Store in history
        image_data = {
            "image": image,
            "embedding": image_embedding,
            "prompt": prompt,
            "tifa_score": tifa_score,
            "prompt_index": prompt_index,
            "timestamp": time.time()
        }
        
        self.selected_images_history.append(image_data)
        
        # Store as reference embedding
        key = f"selected_image_{len(self.selected_images_history)}"
        self.store_embedding(key, image_embedding, "image", {
            "prompt": prompt,
            "tifa_score": tifa_score,
            "is_selected": True
        })
        
        # If this is the first image, store as style/character reference
        if len(self.selected_images_history) == 1:
            self.style_memory = image_embedding
            self.character_memory = image_embedding
            print("🎨 Stored first image as style/character reference")
        
        print(f"🧠 Stored selected image embedding (Total: {len(self.selected_images_history)})")
    
    def calculate_consistency_score(self, image_embedding):
        """Calculate consistency score with previous selected images"""
        if not self.available or not self.selected_images_history or image_embedding is None:
            return 0.5
        
        # Calculate average similarity to all previous selected images
        similarities = []
        for prev_image_data in self.selected_images_history:
            prev_embedding = prev_image_data["embedding"]
            similarity = np.dot(image_embedding.flatten(), prev_embedding.flatten()) / (
                np.linalg.norm(image_embedding) * np.linalg.norm(prev_embedding)
            )
            similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    def calculate_style_consistency(self, image_embedding):
        """Calculate style consistency with first reference image"""
        if not self.available or self.style_memory is None or image_embedding is None:
            return 0.5
        
        similarity = np.dot(image_embedding.flatten(), self.style_memory.flatten()) / (
            np.linalg.norm(image_embedding) * np.linalg.norm(self.style_memory)
        )
        return float(similarity)
    
    def get_similarity_to_memory(self, current_embedding, memory_key):
        """Calculate similarity to stored embedding"""
        if memory_key not in self.memory_bank or current_embedding is None:
            return 0.5
        
        stored_embedding = self.memory_bank[memory_key]["embedding"]
        try:
            similarity = np.dot(current_embedding.flatten(), stored_embedding.flatten()) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
            )
            return float(similarity)
        except:
            return 0.5
    
    def find_most_consistent_image(self, images, reference_prompt, use_consistency_weight=True, consistency_weight=0.3):
        """Find image most consistent with reference prompt AND previous selections"""
        if not self.available:
            return 0, images[0], {}
        
        reference_embedding = self.get_text_embedding(reference_prompt)
        if reference_embedding is None:
            return 0, images[0], {}
        
        best_idx = 0
        best_combined_score = -1
        detailed_scores = []
        
        for idx, image in enumerate(images):
            image_embedding = self.get_image_embedding(image)
            if image_embedding is None:
                detailed_scores.append({
                    "prompt_similarity": 0.0,
                    "consistency_score": 0.0,
                    "style_consistency": 0.0,
                    "combined_score": 0.0
                })
                continue
            
            # Text-image similarity (primary)
            prompt_similarity = np.dot(reference_embedding.flatten(), image_embedding.flatten()) / (
                np.linalg.norm(reference_embedding) * np.linalg.norm(image_embedding)
            )
            
            # Consistency with previous selections
            consistency_score = self.calculate_consistency_score(image_embedding)
            
            # Style consistency with first image
            style_consistency = self.calculate_style_consistency(image_embedding)
            
            # Combined score
            if use_consistency_weight and len(self.selected_images_history) > 0:
                combined_score = (
                    prompt_similarity * 0.5 + 
                    consistency_score * 0.35 +
                    style_consistency * 0.15
                )
            else:
                combined_score = prompt_similarity
            
            scores = {
                "prompt_similarity": float(prompt_similarity),
                "consistency_score": float(consistency_score),
                "style_consistency": float(style_consistency),
                "combined_score": float(combined_score)
            }
            detailed_scores.append(scores)
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_idx = idx
        
        return best_idx, images[best_idx], detailed_scores[best_idx]
    
    def get_consistency_report(self):
        """Get report on consistency across all selected images"""
        if not self.available or len(self.selected_images_history) < 2:
            return {"available": False, "message": "Need at least 2 images for consistency analysis"}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(self.selected_images_history)):
            for j in range(i + 1, len(self.selected_images_history)):
                emb1 = self.selected_images_history[i]["embedding"]
                emb2 = self.selected_images_history[j]["embedding"]
                similarity = np.dot(emb1.flatten(), emb2.flatten()) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                similarities.append(similarity)
        
        return {
            "available": True,
            "total_images": len(self.selected_images_history),
            "average_consistency": float(np.mean(similarities)),
            "min_consistency": float(np.min(similarities)),
            "max_consistency": float(np.max(similarities)),
            "consistency_std": float(np.std(similarities)),
            "consistency_grade": (
                "Excellent" if np.mean(similarities) > 0.8 else
                "Good" if np.mean(similarities) > 0.7 else
                "Moderate" if np.mean(similarities) > 0.6 else
                "Poor"
            )
        }

class ImageCaptioner:
    """Generate captions for TIFA evaluation"""
    
    def __init__(self):
        print("📝 Loading image captioner...")
        
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            self.available = True
            print("✅ Image captioner loaded")
            
        except Exception as e:
            print(f"⚠️ Image captioner unavailable: {e}")
            self.available = False
    
    def generate_caption(self, image):
        """Generate caption for image"""
        if not self.available:
            return "Caption generation unavailable"
        
        try:
            inputs = self.processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Caption generation failed"

class CaptionConsistencyAnalyzer:
    """Compare BLIP captions to detect character inconsistencies and artifacts"""
    
    def __init__(self):
        print("📝 Loading Caption Consistency Analyzer...")
        
        # Store character traits from selected images
        self.character_traits = {
            "hair_color": None,
            "hair_style": None, 
            "clothing": [],
            "age_descriptors": [],
            "physical_features": [],
            "established_scene_elements": []
        }
        
        self.caption_history = []
        self.available = True
        print("✅ Caption Consistency Analyzer ready")
    
    def extract_character_features(self, caption):
        """Extract character features from BLIP caption using NLP"""
        features = {
            "hair_color": [],
            "hair_style": [],
            "clothing": [],
            "age_descriptors": [],
            "physical_features": [],
            "potential_artifacts": [],
            "scene_elements": []
        }
        
        caption_lower = caption.lower()
        
        # Hair colors
        hair_colors = ["brown", "black", "blonde", "red", "gray", "white", "dark", "light"]
        for color in hair_colors:
            if f"{color} hair" in caption_lower or f"{color}-haired" in caption_lower:
                features["hair_color"].append(color)
        
        # Hair styles  
        hair_styles = ["short", "long", "curly", "straight", "messy", "neat"]
        for style in hair_styles:
            if f"{style} hair" in caption_lower:
                features["hair_style"].append(style)
        
        # Clothing
        clothing_items = ["shirt", "uniform", "jacket", "sweater", "pajamas", "clothes", "t-shirt", "dress"]
        for item in clothing_items:
            if item in caption_lower:
                features["clothing"].append(item)
        
        # Age descriptors
        age_words = ["young", "little", "small", "child", "kid", "boy", "teenager"]
        for age in age_words:
            if age in caption_lower:
                features["age_descriptors"].append(age)
        
        # Physical features
        physical = ["glasses", "smile", "eyes", "face", "tall", "short"]
        for feature in physical:
            if feature in caption_lower:
                features["physical_features"].append(feature)
        
        # Scene elements
        scene_words = ["kitchen", "classroom", "cafeteria", "bedroom", "table", "desk", "bed"]
        for scene in scene_words:
            if scene in caption_lower:
                features["scene_elements"].append(scene)
        
        # Potential artifacts (things that shouldn't be there)
        artifacts = ["blurry", "distorted", "multiple", "two heads", "extra", "deformed", 
                    "weird", "strange", "duplicate", "floating", "cut off", "many people",
                    "crowd", "several", "group"]
        for artifact in artifacts:
            if artifact in caption_lower:
                features["potential_artifacts"].append(artifact)
        
        return features
    
    def update_character_profile(self, caption, is_first_image=False):
        """Update character profile from selected image"""
        features = self.extract_character_features(caption)
        
        if is_first_image:
            # Establish character traits from first image
            if features["hair_color"]:
                self.character_traits["hair_color"] = features["hair_color"][0]
                print(f"🎭 Established hair color: {features['hair_color'][0]}")
            
            if features["hair_style"]:
                self.character_traits["hair_style"] = features["hair_style"][0]
                print(f"🎭 Established hair style: {features['hair_style'][0]}")
            
            self.character_traits["clothing"] = features["clothing"]
            self.character_traits["age_descriptors"] = features["age_descriptors"]
            self.character_traits["physical_features"] = features["physical_features"]
            
            print(f"🎭 Character profile established from: {caption}")
        
        # Store caption in history
        self.caption_history.append({
            "caption": caption,
            "features": features,
            "is_reference": is_first_image
        })
    
    def analyze_consistency_and_artifacts(self, current_caption):
        """Analyze current caption for consistency issues and artifacts"""
        current_features = self.extract_character_features(current_caption)
        
        issues = {
            "character_inconsistencies": [],
            "artifacts_detected": [],
            "scene_issues": [],
            "severity": "low"  # low, medium, high
        }
        
        # Check for artifacts first
        if current_features["potential_artifacts"]:
            issues["artifacts_detected"] = current_features["potential_artifacts"]
            issues["severity"] = "high"
            print(f"🚨 Artifacts detected: {', '.join(current_features['potential_artifacts'])}")
        
        # Check character consistency (only if we have established traits)
        if self.character_traits["hair_color"] and current_features["hair_color"]:
            established_hair = self.character_traits["hair_color"]
            current_hair = current_features["hair_color"][0]
            
            if established_hair != current_hair:
                issues["character_inconsistencies"].append(
                    f"Hair color changed from {established_hair} to {current_hair}"
                )
                if issues["severity"] == "low":
                    issues["severity"] = "medium"
                print(f"⚠️ Hair color inconsistency: {established_hair} → {current_hair}")
        
        # Check for multiple people (should be single character)
        multi_person_indicators = ["multiple", "two", "several", "group", "crowd", "many people"]
        for indicator in multi_person_indicators:
            if indicator in current_caption.lower():
                issues["scene_issues"].append(f"Multiple people detected: {indicator}")
                if issues["severity"] == "low":
                    issues["severity"] = "medium"
                print(f"⚠️ Multiple people detected: {indicator}")
        
        return issues
    
    def generate_consistency_improvements(self, issues, original_prompt):
        """Generate specific prompt improvements based on detected issues"""
        improvements = {
            "positive_additions": [],
            "negative_additions": [],
            "specific_fixes": []
        }
        
        # Fix character inconsistencies
        for inconsistency in issues["character_inconsistencies"]:
            if "hair color" in inconsistency.lower():
                if self.character_traits["hair_color"]:
                    improvements["positive_additions"].append(f"{self.character_traits['hair_color']} hair")
                    improvements["specific_fixes"].append(f"Enforce {self.character_traits['hair_color']} hair")
        
        # Fix artifacts
        for artifact in issues["artifacts_detected"]:
            improvements["negative_additions"].append(artifact)
            improvements["specific_fixes"].append(f"Remove {artifact}")
        
        # Fix multiple people issues
        for scene_issue in issues["scene_issues"]:
            if "multiple" in scene_issue.lower() or "people" in scene_issue.lower():
                improvements["positive_additions"].append("single person, one character only")
                improvements["negative_additions"].extend(["multiple people", "crowd", "group", "several people"])
                improvements["specific_fixes"].append("Ensure single character focus")
        
        # Add character consistency terms
        if self.character_traits["hair_color"]:
            improvements["positive_additions"].append(f"consistent {self.character_traits['hair_color']}-haired character")
        
        improvements["positive_additions"].append("same character, consistent design")
        
        return improvements

class PromptImprover:
    """Improve prompts based on CLIP/BLIP analysis with caption consistency"""
    
    def __init__(self, clip_memory, captioner):
        self.clip_memory = clip_memory
        self.captioner = captioner
        self.caption_analyzer = CaptionConsistencyAnalyzer()
        
    def analyze_prompt_image_alignment(self, image, prompt, is_first_image=False):
        """Analyze how well image matches prompt with consistency checking"""
        # Get caption
        caption = self.captioner.generate_caption(image)
        
        # Calculate CLIP similarity
        clip_similarity = 0.5
        if self.clip_memory.available:
            prompt_embedding = self.clip_memory.get_text_embedding(prompt)
            image_embedding = self.clip_memory.get_image_embedding(image)
            
            if prompt_embedding is not None and image_embedding is not None:
                clip_similarity = np.dot(prompt_embedding.flatten(), image_embedding.flatten()) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(image_embedding)
                )
        
        # Update character profile and analyze consistency
        self.caption_analyzer.update_character_profile(caption, is_first_image)
        consistency_issues = self.caption_analyzer.analyze_consistency_and_artifacts(caption)
        
        return {
            "caption": caption,
            "clip_similarity": float(clip_similarity),
            "alignment_quality": "good" if clip_similarity > 0.7 else "moderate" if clip_similarity > 0.5 else "poor",
            "consistency_issues": consistency_issues,
            "character_established": is_first_image
        }
    
    def improve_prompts(self, original_prompt, analysis_result, current_negative=""):
        """Improve positive and negative prompts based on analysis and consistency"""
        improved_positive = original_prompt
        improved_negative = current_negative or "blurry, low quality, distorted, deformed"
        
        improvements = []
        
        # Handle consistency issues first (most important)
        consistency_issues = analysis_result.get("consistency_issues", {})
        if consistency_issues["severity"] != "low":
            consistency_fixes = self.caption_analyzer.generate_consistency_improvements(
                consistency_issues, original_prompt
            )
            
            # Add consistency-specific improvements
            for addition in consistency_fixes["positive_additions"]:
                if addition not in improved_positive.lower():
                    improved_positive = f"{improved_positive}, {addition}"
                    improvements.append(f"Consistency fix: {addition}")
            
            for negative_addition in consistency_fixes["negative_additions"]:
                if negative_addition not in improved_negative.lower():
                    improved_negative = f"{improved_negative}, {negative_addition}"
                    improvements.append(f"Negative consistency: {negative_addition}")
            
            # Log specific fixes
            for fix in consistency_fixes["specific_fixes"]:
                improvements.append(f"Applied: {fix}")
        
        # Standard CLIP-based improvements
        if analysis_result["clip_similarity"] < 0.6:
            clarity_terms = ["detailed", "clear", "well-defined", "accurate representation"]
            for term in clarity_terms:
                if term not in improved_positive.lower():
                    improved_positive = f"{improved_positive}, {term}"
                    improvements.append(f"Added clarity: {term}")
                    break
        
        # Add quality terms if not present
        quality_terms = ["high quality", "masterpiece", "best quality"]
        has_quality = any(term in improved_positive.lower() for term in quality_terms)
        if not has_quality:
            improved_positive = f"{improved_positive}, high quality, detailed"
            improvements.append("Added quality terms")
        
        # Improve negative prompt with standard terms
        common_negatives = ["blurry", "low quality", "distorted", "deformed", "ugly", "bad anatomy"]
        for neg_term in common_negatives:
            if neg_term not in improved_negative.lower():
                improved_negative = f"{improved_negative}, {neg_term}"
                improvements.append(f"Enhanced negative: {neg_term}")
        
        return {
            "positive": improved_positive,
            "negative": improved_negative,
            "improvements": improvements,
            "consistency_issues_found": consistency_issues["severity"] != "low"
        }

class StreamlinedRealCartoonXL:
    """Streamlined Real Cartoon XL v7 with TIFA and CLIP consistency"""
    
    def __init__(self, model_path):
        print("🎨 Loading Streamlined Real Cartoon XL v7...")
        
        self.model_path = model_path
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            self.available = False
            return
        
        try:
            # Load pipeline
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=torch.float16, use_safetensors=True
            )
            
            self.pipe = self.pipe.to("cuda")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            
            # Initialize components
            self.tifa_metric = TIFAMetric()
            self.clip_memory = CLIPMemoryManager()
            self.captioner = ImageCaptioner()
            self.prompt_improver = PromptImprover(self.clip_memory, self.captioner)
            
            self.available = True
            print("✅ Streamlined Real Cartoon XL v7 ready!")
            
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            self.available = False
    
    def generate_with_tifa_selection(self, prompt, negative_prompt="", num_images=3, use_consistency=True, **kwargs):
        """Generate multiple images and select best based on TIFA score and consistency"""
        if not self.available:
            return None
        
        print(f"🎨 Generating {num_images} images for TIFA + consistency selection...")
        
        # Default generation settings
        settings = {
            'prompt': prompt,
            'negative_prompt': negative_prompt or "blurry, low quality, distorted, deformed",
            'height': 1024,
            'width': 1024,
            'num_inference_steps': 30,
            'guidance_scale': 5.0,
            'num_images_per_prompt': num_images,
            **kwargs
        }
        
        # Generate images
        start_time = time.time()
        result = self.pipe(**settings)
        gen_time = time.time() - start_time
        
        # Evaluate each image with TIFA and consistency
        image_evaluations = []
        is_first_prompt = len(self.clip_memory.selected_images_history) == 0
        
        for idx, image in enumerate(result.images):
            # Generate caption
            caption = self.captioner.generate_caption(image)
            
            # Calculate TIFA score
            tifa_result = self.tifa_metric.calculate_tifa_score(image, prompt, caption)
            tifa_score = tifa_result["score"]
            
            # Calculate consistency scores only if we have previous images AND not first prompt
            consistency_scores = {}
            if use_consistency and self.clip_memory.available and not is_first_prompt:
                image_embedding = self.clip_memory.get_image_embedding(image)
                consistency_scores = {
                    "overall_consistency": self.clip_memory.calculate_consistency_score(image_embedding),
                    "style_consistency": self.clip_memory.calculate_style_consistency(image_embedding)
                }
            
            evaluation = {
                "idx": idx,
                "image": image,
                "caption": caption,
                "tifa_score": tifa_score,
                "consistency_scores": consistency_scores
            }
            image_evaluations.append(evaluation)
            
            # Display scores - only show consistency if not first prompt
            if is_first_prompt:
                print(f"   Image {idx+1}: TIFA = {tifa_score:.3f}")
            else:
                consistency_info = ""
                if consistency_scores:
                    consistency_info = f", Consistency = {consistency_scores['overall_consistency']:.3f}"
                print(f"   Image {idx+1}: TIFA = {tifa_score:.3f}{consistency_info}")
        
        # Select best image based on combined TIFA + consistency score
        best_evaluation = None
        best_combined_score = -1
        
        for evaluation in image_evaluations:
            tifa_score = evaluation["tifa_score"]
            
            # Combine TIFA with consistency only if we have previous images AND consistency was calculated
            if use_consistency and evaluation["consistency_scores"] and not is_first_prompt:
                consistency_weight = 0.5  # 50% weight for consistency
                consistency_score = evaluation["consistency_scores"]["overall_consistency"]
                combined_score = tifa_score * (1 - consistency_weight) + consistency_score * consistency_weight
            else:
                combined_score = tifa_score
            
            evaluation["combined_score"] = combined_score
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_evaluation = evaluation
        
        best_image = best_evaluation["image"]
        best_idx = best_evaluation["idx"]
        best_tifa = best_evaluation["tifa_score"]
        
        # Store selected image in CLIP memory for future consistency
        if self.clip_memory.available:
            self.clip_memory.store_selected_image(
                best_image, 
                prompt, 
                best_tifa,
                prompt_index=len(self.clip_memory.selected_images_history)
            )
        
        # Display selection result - only show consistency if not first prompt
        if is_first_prompt:
            print(f"✅ Best image: #{best_idx+1} with TIFA = {best_tifa:.3f}")
        else:
            consistency_info = ""
            if best_evaluation["consistency_scores"]:
                cons_score = best_evaluation["consistency_scores"]["overall_consistency"]
                consistency_info = f", Consistency = {cons_score:.3f}"
            print(f"✅ Best image: #{best_idx+1} with TIFA = {best_tifa:.3f}{consistency_info}")
        
        print(f"⏱️ Generation time: {gen_time:.2f}s")
        
        return {
            "best_image": best_image,
            "best_idx": best_idx,
            "best_tifa_score": best_tifa,
            "best_combined_score": best_combined_score,
            "all_evaluations": image_evaluations,
            "all_images": result.images,
            "generation_time": gen_time,
            "prompt": prompt,
            "negative_prompt": settings['negative_prompt'],
            "consistency_used": use_consistency and not is_first_prompt
        }
    
    def iterative_improvement_generation(self, initial_prompt, iterations=3, images_per_iteration=3):
        """Generate with iterative prompt improvement"""
        print(f"🔄 Starting iterative improvement generation...")
        print(f"📝 Initial prompt: {initial_prompt}")
        
        current_prompt = initial_prompt
        current_negative = "blurry, low quality, distorted, deformed"
        
        results_history = []
        
        for iteration in range(iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            print(f"📝 Current prompt: {current_prompt}")
            print(f"🚫 Current negative: {current_negative}")
            
            # Generate images with current prompts
            result = self.generate_with_tifa_selection(
                current_prompt, 
                current_negative, 
                num_images=images_per_iteration
            )
            
            if result is None:
                break
            
            # Analyze the best image with consistency checking
            is_first_selected = len(self.clip_memory.selected_images_history) == 0
            analysis = self.prompt_improver.analyze_prompt_image_alignment(
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
                "best_image": result["best_image"],
                "tifa_score": result["best_tifa_score"],
                "clip_similarity": analysis["clip_similarity"],
                "caption": analysis["caption"],
                "improvements_applied": []
            }
            
            # Improve prompts for next iteration (if not last iteration)
            if iteration < iterations - 1:
                improved = self.prompt_improver.improve_prompts(
                    current_prompt, 
                    analysis, 
                    current_negative
                )
                
                current_prompt = improved["positive"]
                current_negative = improved["negative"]
                iteration_result["improvements_applied"] = improved["improvements"]
                
                print(f"🚀 Applied improvements: {', '.join(improved['improvements'])}")
            
            results_history.append(iteration_result)
        
        # THIS WAS THE MISSING PART! Find best result and return it
        if not results_history:
            print("❌ No successful iterations")
            return None
        
        # Find best result across all iterations
        best_iteration = max(results_history, key=lambda x: x["tifa_score"])
        
        print(f"\n🏆 ITERATIVE IMPROVEMENT COMPLETE")
        print(f"📈 Score progression: {[f'{r['tifa_score']:.3f}' for r in results_history]}")
        print(f"🥇 Best iteration: {best_iteration['iteration']} (TIFA: {best_iteration['tifa_score']:.3f})")
        print(f"📊 Best CLIP similarity: {best_iteration['clip_similarity']:.3f}")
        
        final_result = {
            "best_result": best_iteration,
            "history": results_history,
            "initial_prompt": initial_prompt,
            "final_prompt": results_history[-1]["prompt"],
            "score_improvement": best_iteration["tifa_score"] - results_history[0]["tifa_score"]
        }
        
        return final_result
            
    def generate_storyboard_sequence(self, prompts_list, iterations_per_prompt=2, images_per_iteration=3):
        """Generate a sequence of images with consistency across prompts"""
        print(f"📚 Generating storyboard sequence with {len(prompts_list)} prompts")
        print(f"🔄 {iterations_per_prompt} iterations per prompt, {images_per_iteration} images per iteration")
        
        storyboard_results = []
        
        for prompt_idx, prompt in enumerate(prompts_list):
            print(f"\n📖 PROMPT {prompt_idx + 1}/{len(prompts_list)}: {prompt}")
            
            # Generate with iterative improvement for this prompt
            result = self.iterative_improvement_generation(
                prompt, 
                iterations=iterations_per_prompt, 
                images_per_iteration=images_per_iteration
            )
            
            # Debug: Check what we got back
            print(f"🔍 Debug: Result type = {type(result)}")
            if result is not None:
                print(f"🔍 Debug: Result keys = {list(result.keys())}")
                print(f"🔍 Debug: Best result type = {type(result.get('best_result'))}")
            
            # Safety check
            if result is None:
                print(f"❌ Failed to generate result for prompt {prompt_idx + 1}")
                continue
            
            # Add prompt index to result
            result["prompt_index"] = prompt_idx
            result["prompt_text"] = prompt
            
            storyboard_results.append(result)
            
            # Show consistency progress
            if prompt_idx > 0 and self.clip_memory.available:
                consistency_report = self.clip_memory.get_consistency_report()
                if consistency_report["available"]:
                    print(f"📊 Storyboard Consistency: {consistency_report['consistency_grade']} "
                          f"(Avg: {consistency_report['average_consistency']:.3f})")
        
        # Final consistency report
        final_consistency = self.clip_memory.get_consistency_report()
        
        print(f"\n🎉 STORYBOARD SEQUENCE COMPLETE!")
        print(f"📖 Total prompts: {len(prompts_list)}")
        print(f"🎨 Total images generated: {len(prompts_list) * iterations_per_prompt * images_per_iteration}")
        print(f"🏆 Selected images: {len(storyboard_results)}")
        
        if final_consistency["available"]:
            print(f"📊 Overall Consistency: {final_consistency['consistency_grade']} "
                  f"(Score: {final_consistency['average_consistency']:.3f})")
            print(f"   Range: {final_consistency['min_consistency']:.3f} - {final_consistency['max_consistency']:.3f}")
        
        # Calculate TIFA progression
        tifa_scores = [result["best_result"]["tifa_score"] for result in storyboard_results]
        print(f"🎯 TIFA Scores: {[f'{score:.3f}' for score in tifa_scores]}")
        print(f"📈 Average TIFA: {np.mean(tifa_scores):.3f}")
        
        return {
            "storyboard_results": storyboard_results,
            "consistency_report": final_consistency,
            "tifa_scores": tifa_scores,
            "average_tifa": float(np.mean(tifa_scores)),
            "prompts_list": prompts_list,
            "generation_summary": {
                "total_prompts": len(prompts_list),
                "iterations_per_prompt": iterations_per_prompt,
                "images_per_iteration": images_per_iteration,
                "total_images_generated": len(prompts_list) * iterations_per_prompt * images_per_iteration,
                "selected_images": len(storyboard_results)
            }
        }

def demo_streamlined_system():
    """Demo the streamlined system"""
    print("\n🎨 STREAMLINED REAL CARTOON XL v7 DEMO")
    print("=" * 50)
    print("🎯 Core Features:")
    print("✅ Real Cartoon XL v7 model")
    print("✅ TIFA scoring for best image selection")
    print("✅ CLIP embeddings for consistency across prompts")
    print("✅ Generate 3 images, pick best 50% TIFA + 50% consistency")
    print("✅ Iterative prompt improvement")
    print("✅ BLIP/CLIP analysis for enhancement")
    print("✅ Cross-prompt consistency tracking") 
    print("✅ Caption-based consistency analysis")
    print("✅ Automatic artifact detection")
    print("✅ Character trait consistency enforcement")
    print("✅ Storyboard sequence generation")
    
    print("\n🧠 ENHANCED CONSISTENCY FEATURES:")
    print("✅ Stores selected image embeddings")
    print("✅ Tracks style consistency with first image")
    print("✅ Calculates consistency scores across all images")
    print("✅ Combines 50% TIFA + 50% consistency for selection")
    print("✅ BLIP caption comparison for character traits")
    print("✅ Automatic artifact detection (blurry, multiple people, etc.)")
    print("✅ Character trait enforcement (hair color, clothing, etc.)")
    print("✅ Specific prompt fixes based on detected issues")
    print("✅ Provides detailed consistency reports and grades")
    
    print("\n💡 USAGE:")
    print("# Single prompt with iterations")
    print("generator = StreamlinedRealCartoonXL('path/to/realcartoonxl_v7.safetensors')")
    print("result = generator.iterative_improvement_generation('boy brushing teeth')")
    print()
    print("# Multi-prompt storyboard with consistency")
    print("prompts = ['boy waking up', 'boy brushing teeth', 'boy eating breakfast']")
    print("storyboard = generator.generate_storyboard_sequence(prompts)")
    print("# Automatically maintains character/style consistency across prompts")
    
    print("\n🔄 CONSISTENCY PROCESS:")
    print("1. First image establishes style/character reference")
    print("2. Subsequent images scored on TIFA + consistency")
    print("3. CLIP embeddings track visual consistency")
    print("4. System balances 50% prompt faithfulness + 50% consistency")
    print("5. Consistency reports show overall coherence")
    print("6. Perfect for character-consistent storyboards")

if __name__ == "__main__":
    demo_streamlined_system()

    #outputted to :daily_routine_20250604_214100