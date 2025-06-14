"""
Quality Evaluator - Handles TIFA scoring and image quality assessment
Extracted from StreamlinedRealCartoonXL system for modular architecture
"""

import torch
import numpy as np
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, 
    CLIPProcessor, CLIPModel
)
from sentence_transformers import SentenceTransformer
try:
    import spacy
except ImportError:
    spacy = None


class QualityEvaluator:
    """Evaluates image quality using TIFA scoring and artifact detection"""
    
    def __init__(self):
        print("🎯 Loading Quality Evaluator...")
        
        try:
            # CLIP for visual-semantic alignment
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            
            # BLIP for image captioning
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            
            # NLP model for semantic parsing
            try:
                if spacy is not None:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.spacy_available = True
                    print("✅ SpaCy NLP loaded")
                else:
                    raise ImportError("SpaCy not installed")
            except:
                print("⚠️ SpaCy not available - using basic parsing")
                self.spacy_available = False
            
            # Sentence transformer for semantic embeddings
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
                self.blip_model = self.blip_model.to("cuda")
            
            # Configuration
            self.tifa_weights = {
                "clip_weight": 0.8,
                "semantic_weight": 0.2
            }
            
            self.available = True
            print("✅ Quality Evaluator loaded")
            
        except Exception as e:
            print(f"⚠️ Quality Evaluator unavailable: {e}")
            self.available = False
    
    def generate_caption(self, image):
        """Generate caption for image using BLIP"""
        if not self.available:
            return "Caption generation unavailable"
        
        try:
            inputs = self.blip_processor(image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=50)
            
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Caption generation failed"
    
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
    
    def detect_artifacts(self, caption):
        """Detect visual artifacts and quality issues from caption"""
        artifacts = {
            "quality_issues": [],
            "visual_artifacts": [],
            "character_issues": [],
            "severity": "low"
        }
        
        caption_lower = caption.lower()
        
        # Quality issues
        quality_indicators = ["blurry", "pixelated", "low quality", "distorted", "unclear", "fuzzy"]
        for indicator in quality_indicators:
            if indicator in caption_lower:
                artifacts["quality_issues"].append(indicator)
                artifacts["severity"] = "high"
        
        # Visual artifacts
        visual_indicators = ["deformed", "extra", "duplicate", "floating", "cut off", "weird", "strange"]
        for indicator in visual_indicators:
            if indicator in caption_lower:
                artifacts["visual_artifacts"].append(indicator)
                if artifacts["severity"] == "low":
                    artifacts["severity"] = "medium"
        
        # Character issues
        character_indicators = ["multiple", "two heads", "many people", "crowd", "several", "group"]
        for indicator in character_indicators:
            if indicator in caption_lower:
                artifacts["character_issues"].append(indicator)
                if artifacts["severity"] == "low":
                    artifacts["severity"] = "medium"
        
        return artifacts
    
    def calculate_tifa_score(self, image, prompt, generated_caption=None):
        """Calculate TIFA (Text-Image Faithfulness Assessment) score"""
        if not self.available:
            return {"score": 0.5, "details": "TIFA unavailable"}
        
        try:
            # Generate caption if not provided
            if generated_caption is None:
                generated_caption = self.generate_caption(image)
            
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
            
            # Detect artifacts
            artifacts = self.detect_artifacts(generated_caption)
            
            # Final score calculation
            clip_score = img_prompt_sim
            semantic_score = np.mean(list(semantic_overlaps.values())) if semantic_overlaps else 0.5
            
            # Apply artifact penalty
            artifact_penalty = 0.0
            if artifacts["severity"] == "high":
                artifact_penalty = 0.3
            elif artifacts["severity"] == "medium":
                artifact_penalty = 0.15
            
            # Weighted combination with artifact penalty
            tifa_score = (
                (clip_score * self.tifa_weights["clip_weight"]) + 
                (semantic_score * self.tifa_weights["semantic_weight"])
            ) * (1.0 - artifact_penalty)
            
            return {
                "score": float(max(0.0, min(1.0, tifa_score))),  # Clamp to [0,1]
                "details": {
                    "clip_score": float(clip_score),
                    "semantic_score": float(semantic_score),
                    "component_overlaps": semantic_overlaps,
                    "image_prompt_similarity": float(img_prompt_sim),
                    "prompt_caption_similarity": float(prompt_caption_sim),
                    "artifacts_detected": artifacts,
                    "artifact_penalty": artifact_penalty,
                    "generated_caption": generated_caption
                }
            }
            
        except Exception as e:
            print(f"TIFA calculation error: {e}")
            return {"score": 0.5, "details": f"Error: {e}"}
    
    def evaluate_batch(self, images, prompt):
        """Evaluate a batch of images and return scores"""
        if not self.available:
            return [{"score": 0.5, "details": "Evaluator unavailable"} for _ in images]
        
        results = []
        for idx, image in enumerate(images):
            result = self.calculate_tifa_score(image, prompt)
            result["image_index"] = idx
            results.append(result)
            
            # Log result
            score = result["score"]
            artifacts = result["details"].get("artifacts_detected", {})
            artifact_info = f" (Artifacts: {artifacts['severity']})" if artifacts["severity"] != "low" else ""
            print(f"   Image {idx+1}: TIFA = {score:.3f}{artifact_info}")
        
        return results
    
    def update_tifa_weights(self, clip_weight=None, semantic_weight=None):
        """Update TIFA scoring weights"""
        if clip_weight is not None:
            self.tifa_weights["clip_weight"] = clip_weight
        if semantic_weight is not None:
            self.tifa_weights["semantic_weight"] = semantic_weight
        
        # Normalize weights
        total = sum(self.tifa_weights.values())
        for key in self.tifa_weights:
            self.tifa_weights[key] /= total
        
        print(f"🎯 Updated TIFA weights: {self.tifa_weights}")
    
    def get_quality_summary(self, evaluation_results):
        """Get summary statistics from batch evaluation"""
        if not evaluation_results:
            return {}
        
        scores = [r["score"] for r in evaluation_results]
        artifacts = [r["details"].get("artifacts_detected", {}) for r in evaluation_results]
        
        high_artifact_count = sum(1 for a in artifacts if a.get("severity") == "high")
        medium_artifact_count = sum(1 for a in artifacts if a.get("severity") == "medium")
        
        return {
            "average_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "score_std": float(np.std(scores)),
            "high_artifact_images": high_artifact_count,
            "medium_artifact_images": medium_artifact_count,
            "clean_images": len(scores) - high_artifact_count - medium_artifact_count,
            "quality_grade": (
                "Excellent" if np.mean(scores) > 0.8 else
                "Good" if np.mean(scores) > 0.7 else
                "Moderate" if np.mean(scores) > 0.6 else
                "Poor"
            )
        }