"""
Consistency Manager - Handles CLIP-based consistency tracking and IP-Adapter preparation
Extracted from StreamlinedRealCartoonXL system for modular architecture
Complete implementation with all original functionality
"""

import torch
import numpy as np
import time
from transformers import CLIPProcessor, CLIPModel


class ConsistencyManager:
    """Manages visual consistency across image sequences using CLIP embeddings"""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        print("🧠 Loading CLIP memory manager...")
        
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
            
            # Memory storage - exactly as in original
            self.memory_bank = {}  # Store embeddings for consistency
            self.selected_images_history = []  # Track selected images across prompts
            self.style_memory = None  # Store style embedding from first image
            self.character_memory = None  # Store character embedding
            
            # Configuration - matching original weights
            self.consistency_weights = {
                "prompt_similarity": 0.5,
                "overall_consistency": 0.35,
                "style_consistency": 0.15
            }
            
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
        
        # Store in history - exactly as in original
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
        
        return image_embedding  # Return for potential IP-Adapter use
    
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
            
            # Combined score - exactly as in original
            if use_consistency_weight and len(self.selected_images_history) > 0:
                combined_score = (
                    prompt_similarity * self.consistency_weights["prompt_similarity"] + 
                    consistency_score * self.consistency_weights["overall_consistency"] +
                    style_consistency * self.consistency_weights["style_consistency"]
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
        
        # Calculate pairwise similarities - exactly as in original
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
    
    def get_reference_image_for_ip_adapter(self):
        """Get the best reference image for IP-Adapter (usually first or highest quality)"""
        if not self.selected_images_history:
            return None
        
        # Return first image (character reference) by default
        # Could be enhanced to return highest quality image
        return self.selected_images_history[0]["image"]
    
    def get_character_embedding(self):
        """Get character embedding for potential IP-Adapter use"""
        return self.character_memory
    
    def get_style_embedding(self):
        """Get style embedding for potential IP-Adapter use"""
        return self.style_memory
    
    def update_consistency_weights(self, prompt_sim=None, overall_cons=None, style_cons=None):
        """Update consistency weighting for different use cases"""
        if prompt_sim is not None:
            self.consistency_weights["prompt_similarity"] = prompt_sim
        if overall_cons is not None:
            self.consistency_weights["overall_consistency"] = overall_cons
        if style_cons is not None:
            self.consistency_weights["style_consistency"] = style_cons
        
        # Normalize weights
        total = sum(self.consistency_weights.values())
        for key in self.consistency_weights:
            self.consistency_weights[key] /= total
        
        print(f"🎯 Updated consistency weights: {self.consistency_weights}")
    
    def reset_memory(self):
        """Reset all stored consistency memory (for new storyboard)"""
        self.memory_bank = {}
        self.selected_images_history = []
        self.style_memory = None
        self.character_memory = None
        print("🔄 Consistency memory reset")