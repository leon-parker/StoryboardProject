"""
Quality Evaluator - Handles TIFA scoring and image quality assessment
Enhanced with computer vision artifact detection
"""

import torch
import numpy as np
import cv2
from PIL import Image
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
    """Evaluates image quality using TIFA scoring and visual artifact detection"""
    
    def __init__(self):
        print("🎯 Loading Quality Evaluator with Computer Vision...")
        
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
            
            # Initialize computer vision components
            self._init_cv_detectors()
            
            # Configuration
            self.tifa_weights = {
                "clip_weight": 0.6,        # Reduced to make room for visual
                "semantic_weight": 0.2,
                "visual_weight": 0.2       # NEW: Visual artifact weight
            }
            
            self.available = True
            print("✅ Quality Evaluator with CV loaded")
            
        except Exception as e:
            print(f"⚠️ Quality Evaluator unavailable: {e}")
            self.available = False
    
    def _init_cv_detectors(self):
        """Initialize computer vision artifact detectors"""
        try:
            # Face detection cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.cv_available = True
            print("✅ Computer vision detectors loaded")
        except Exception as e:
            print(f"⚠️ CV detectors unavailable: {e}")
            self.cv_available = False
    
    def detect_visual_artifacts(self, image):
        """Detect artifacts by analyzing image pixels with computer vision"""
        artifacts = {
            "blur_score": 0.0,
            "brightness_issues": [],
            "face_anomalies": [],
            "duplicate_regions": [],
            "distortion_score": 0.0,
            "color_anomalies": [],
            "edge_quality": 0.0,
            "noise_level": 0.0,
            "visual_severity": "low"
        }
        
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            height, width = gray.shape
            
            # 1. BLUR DETECTION using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_variance = laplacian.var()
            artifacts["blur_score"] = float(blur_variance)
            
            if blur_variance < 100:  # Threshold for blurry images
                artifacts["visual_severity"] = "high"
                print(f"   🔍 Blur detected: {blur_variance:.1f}")
            
            # 2. BRIGHTNESS/DARKNESS ISSUES
            mean_brightness = np.mean(img_array)
            artifacts["brightness_mean"] = float(mean_brightness)
            
            if mean_brightness < 20:  # Very dark
                artifacts["brightness_issues"].append("very_dark_image")
                artifacts["visual_severity"] = "high"
                print(f"   🔍 Very dark image: {mean_brightness:.1f}")
            elif mean_brightness > 240:  # Very bright
                artifacts["brightness_issues"].append("overexposed_image")
                artifacts["visual_severity"] = "medium"
            elif mean_brightness < 50:  # Somewhat dark
                artifacts["brightness_issues"].append("dark_image")
                artifacts["visual_severity"] = "medium" if artifacts["visual_severity"] == "low" else artifacts["visual_severity"]
            
            # 3. FACE ANOMALY DETECTION
            if self.cv_available:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 1:  # Multiple faces in single character image
                    artifacts["face_anomalies"].append("multiple_faces")
                    artifacts["visual_severity"] = "high"
                    print(f"   🔍 Multiple faces detected: {len(faces)}")
                
                # Check face quality for each detected face
                for i, (x, y, w, h) in enumerate(faces):
                    face_region = gray[y:y+h, x:x+w]
                    
                    # Face blur check
                    face_blur = cv2.Laplacian(face_region, cv2.CV_64F).var()
                    if face_blur < 50:
                        artifacts["face_anomalies"].append(f"blurry_face_{i}")
                        artifacts["visual_severity"] = "medium" if artifacts["visual_severity"] == "low" else artifacts["visual_severity"]
                    
                    # Face size check (too small might be an artifact)
                    face_area = w * h
                    image_area = width * height
                    face_ratio = face_area / image_area
                    
                    if face_ratio < 0.01:  # Very small face
                        artifacts["face_anomalies"].append(f"tiny_face_{i}")
                    elif face_ratio > 0.8:  # Face takes up whole image
                        artifacts["face_anomalies"].append(f"oversized_face_{i}")
            
            # 4. EDGE QUALITY ASSESSMENT
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            artifacts["edge_quality"] = float(edge_density)
            
            if edge_density < 0.02:  # Very few edges
                artifacts["visual_severity"] = "medium" if artifacts["visual_severity"] == "low" else artifacts["visual_severity"]
                print(f"   🔍 Poor edge quality: {edge_density:.3f}")
            
            # 5. NOISE DETECTION
            noise_level = np.std(gray)
            artifacts["noise_level"] = float(noise_level)
            
            if noise_level > 80:  # High noise
                artifacts["visual_severity"] = "medium" if artifacts["visual_severity"] == "low" else artifacts["visual_severity"]
            
            # 6. COLOR ANOMALY DETECTION
            for channel_idx, channel_name in enumerate(['red', 'green', 'blue']):
                channel_data = img_array[:, :, channel_idx]
                channel_mean = np.mean(channel_data)
                channel_std = np.std(channel_data)
                
                # Detect extreme color casts
                if channel_mean < 10:
                    artifacts["color_anomalies"].append(f"missing_{channel_name}_channel")
                elif channel_mean > 245:
                    artifacts["color_anomalies"].append(f"oversaturated_{channel_name}_channel")
                
                # Detect low color variation
                if channel_std < 5:
                    artifacts["color_anomalies"].append(f"low_variation_{channel_name}")
            
            # 7. DISTORTION DETECTION using symmetry analysis
            left_half = gray[:, :width//2]
            right_half = np.fliplr(gray[:, width//2:width//2 + left_half.shape[1]])
            
            if left_half.shape == right_half.shape:
                asymmetry = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
                artifacts["distortion_score"] = float(asymmetry)
                
                if asymmetry > 50:
                    artifacts["visual_severity"] = "medium" if artifacts["visual_severity"] == "low" else artifacts["visual_severity"]
            
            # 8. DUPLICATE REGION DETECTION (simplified)
            block_size = min(32, width//8, height//8)
            if block_size > 8:
                duplicate_count = self._detect_duplicate_blocks(gray, block_size)
                if duplicate_count > 10:
                    artifacts["duplicate_regions"].append("repetitive_patterns")
                    artifacts["visual_severity"] = "medium" if artifacts["visual_severity"] == "low" else artifacts["visual_severity"]
        
        except Exception as e:
            print(f"⚠️ Visual artifact detection error: {e}")
            artifacts["visual_severity"] = "unknown"
        
        return artifacts
    
    def _detect_duplicate_blocks(self, gray_image, block_size):
        """Detect duplicate/repetitive blocks in image"""
        height, width = gray_image.shape
        blocks = []
        
        # Extract blocks
        for i in range(0, height - block_size, block_size//2):
            for j in range(0, width - block_size, block_size//2):
                block = gray_image[i:i+block_size, j:j+block_size]
                blocks.append(block)
        
        # Count similar blocks
        duplicate_count = 0
        threshold = 15
        
        for i, block1 in enumerate(blocks):
            for j, block2 in enumerate(blocks[i+1:], i+1):
                if block1.shape == block2.shape:
                    diff = np.mean(np.abs(block1.astype(float) - block2.astype(float)))
                    if diff < threshold:
                        duplicate_count += 1
        
        return duplicate_count
    
    def detect_artifacts(self, caption, image=None):
        """Enhanced: Combine caption analysis with visual analysis"""
        # Original caption-based detection
        artifacts = {
            "caption_issues": [],
            "visual_issues": [],
            "character_issues": [],
            "severity": "low"
        }
        
        caption_lower = caption.lower()
        
        # Caption-based detection (existing)
        quality_indicators = ["blurry", "pixelated", "low quality", "distorted", "unclear", "fuzzy"]
        for indicator in quality_indicators:
            if indicator in caption_lower:
                artifacts["caption_issues"].append(indicator)
                artifacts["severity"] = "high"
        
        visual_indicators = ["deformed", "extra", "duplicate", "floating", "cut off", "weird", "strange"]
        for indicator in visual_indicators:
            if indicator in caption_lower:
                artifacts["caption_issues"].append(indicator)
                if artifacts["severity"] == "low":
                    artifacts["severity"] = "medium"
        
        character_indicators = ["multiple", "two heads", "many people", "crowd", "several", "group"]
        for indicator in character_indicators:
            if indicator in caption_lower:
                artifacts["character_issues"].append(indicator)
                if artifacts["severity"] == "low":
                    artifacts["severity"] = "medium"
        
        # NEW: Visual analysis if image provided
        if image is not None:
            visual_artifacts = self.detect_visual_artifacts(image)
            artifacts["visual_analysis"] = visual_artifacts
            
            # Update severity based on visual analysis
            visual_severity = visual_artifacts["visual_severity"]
            if visual_severity == "high":
                artifacts["severity"] = "high"
            elif visual_severity == "medium" and artifacts["severity"] == "low":
                artifacts["severity"] = "medium"
            
            # Add visual issues to main list
            if visual_artifacts["blur_score"] < 100:
                artifacts["visual_issues"].append("blur_detected")
            
            if visual_artifacts["brightness_issues"]:
                artifacts["visual_issues"].extend(visual_artifacts["brightness_issues"])
            
            if visual_artifacts["face_anomalies"]:
                artifacts["visual_issues"].extend(visual_artifacts["face_anomalies"])
            
            if visual_artifacts["duplicate_regions"]:
                artifacts["visual_issues"].extend(visual_artifacts["duplicate_regions"])
        
        return artifacts
    
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
    
    def calculate_tifa_score(self, image, prompt, generated_caption=None):
        """Enhanced TIFA score with visual artifact penalties"""
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
            
            # Enhanced: Detect artifacts with visual analysis
            artifacts = self.detect_artifacts(generated_caption, image)
            
            # Calculate base scores
            clip_score = img_prompt_sim
            semantic_score = np.mean(list(semantic_overlaps.values())) if semantic_overlaps else 0.5
            
            # NEW: Visual quality score based on computer vision analysis
            visual_quality_score = 1.0  # Start with perfect score
            
            if "visual_analysis" in artifacts:
                visual_data = artifacts["visual_analysis"]
                
                # Blur penalty
                if visual_data["blur_score"] < 100:
                    blur_penalty = max(0, (100 - visual_data["blur_score"]) / 100)
                    visual_quality_score -= blur_penalty * 0.4
                
                # Brightness penalty
                if visual_data["brightness_issues"]:
                    visual_quality_score -= 0.2 * len(visual_data["brightness_issues"])
                
                # Face anomaly penalty
                if visual_data["face_anomalies"]:
                    visual_quality_score -= 0.3 * len(visual_data["face_anomalies"])
                
                # Edge quality bonus/penalty
                edge_quality = visual_data["edge_quality"]
                if edge_quality < 0.02:  # Very poor edges
                    visual_quality_score -= 0.15
                elif edge_quality > 0.1:  # Good edges
                    visual_quality_score += 0.05
                
                # Color anomaly penalty
                if visual_data["color_anomalies"]:
                    visual_quality_score -= 0.1 * len(visual_data["color_anomalies"])
            
            visual_quality_score = max(0.0, min(1.0, visual_quality_score))  # Clamp to [0,1]
            
            # Apply artifact penalty (existing system)
            artifact_penalty = 0.0
            if artifacts["severity"] == "high":
                artifact_penalty = 0.3
            elif artifacts["severity"] == "medium":
                artifact_penalty = 0.15
            
            # Enhanced: Weighted combination with visual quality
            tifa_score = (
                (clip_score * self.tifa_weights["clip_weight"]) + 
                (semantic_score * self.tifa_weights["semantic_weight"]) +
                (visual_quality_score * self.tifa_weights["visual_weight"])
            ) * (1.0 - artifact_penalty)
            
            return {
                "score": float(max(0.0, min(1.0, tifa_score))),  # Clamp to [0,1]
                "details": {
                    "clip_score": float(clip_score),
                    "semantic_score": float(semantic_score),
                    "visual_quality_score": float(visual_quality_score),
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
            
            # Enhanced logging with visual info
            score = result["score"]
            details = result["details"]
            artifacts = details.get("artifacts_detected", {})
            visual_score = details.get("visual_quality_score", 0.5)
            
            artifact_info = f" (Artifacts: {artifacts['severity']})" if artifacts["severity"] != "low" else ""
            visual_info = f" (Visual: {visual_score:.2f})" if visual_score < 0.9 else ""
            
            print(f"   Image {idx+1}: TIFA = {score:.3f}{artifact_info}{visual_info}")
        
        return results
    
    def update_tifa_weights(self, clip_weight=None, semantic_weight=None, visual_weight=None):
        """Update TIFA scoring weights"""
        if clip_weight is not None:
            self.tifa_weights["clip_weight"] = clip_weight
        if semantic_weight is not None:
            self.tifa_weights["semantic_weight"] = semantic_weight
        if visual_weight is not None:
            self.tifa_weights["visual_weight"] = visual_weight
        
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
        visual_scores = [r["details"].get("visual_quality_score", 0.5) for r in evaluation_results]
        
        high_artifact_count = sum(1 for a in artifacts if a.get("severity") == "high")
        medium_artifact_count = sum(1 for a in artifacts if a.get("severity") == "medium")
        
        return {
            "average_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "score_std": float(np.std(scores)),
            "average_visual_quality": float(np.mean(visual_scores)),
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