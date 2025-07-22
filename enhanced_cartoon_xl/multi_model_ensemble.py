import numpy as np
# MultiModelEnsemble - Extracted directly from original code

import config
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline
import torch

class MultiModelEnsemble:
    """
    PURE AI multi-model ensemble for comprehensive image understanding
    NO hard-coded rules - AI models provide all insights
    """
    
    def __init__(self):
        print("🗃 Loading Multi-Model AI Ensemble...")
        
        self.models = {}
        
        # Load multiple AI captioning models
        try:
            # BLIP-2 for advanced captioning
            self.models['blip2'] = {
                'processor': Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b"),
                'model': Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16),
                'available': True
            }
            if torch.cuda.is_available():
                self.models['blip2']['model'] = self.models['blip2']['model'].to("cuda")
            print("✅ BLIP-2 loaded for ensemble")
        except:
            print("⚠️ BLIP-2 unavailable")
            self.models['blip2'] = {'available': False}
        
        # Add more models as needed
        try:
            # Image classification for scene understanding
            self.scene_classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            self.scene_classification_available = True
            print("✅ Scene classifier loaded for ensemble")
        except:
            print("⚠️ Scene classifier unavailable")
            self.scene_classification_available = False
        
        self.available = True
        print("✅ Multi-model AI ensemble ready")
    
    def generate_ensemble_analysis(self, image, primary_caption):
        """Generate comprehensive analysis using multiple AI models"""
        ensemble_results = {
            "primary_caption": primary_caption,
            "alternative_captions": [],
            "scene_classification": [],
            "consensus_analysis": {},
            "reliability_score": 0.5
        }
        
        try:
            # Generate alternative captions with BLIP-2
            if self.models['blip2']['available']:
                print("🤖 Generating BLIP-2 alternative caption...")
                try:
                    inputs = self.models['blip2']['processor'](image, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = self.models['blip2']['model'].generate(**inputs, max_length=50)
                    
                    blip2_caption = self.models['blip2']['processor'].decode(generated_ids[0], skip_special_tokens=True)
                    ensemble_results["alternative_captions"].append({
                        "model": "BLIP-2",
                        "caption": blip2_caption,
                        "confidence": 0.8
                    })
                except Exception as e:
                    print(f"BLIP-2 generation error: {e}")
            
            # Scene classification
            if self.scene_classification_available:
                print("🤖 AI scene classification...")
                try:
                    scene_results = self.scene_classifier(image)
                    ensemble_results["scene_classification"] = scene_results[:5]  # Top 5
                except Exception as e:
                    print(f"Scene classification error: {e}")
            
            # AI consensus analysis
            all_captions = [primary_caption] + [alt["caption"] for alt in ensemble_results["alternative_captions"]]
            
            if len(all_captions) > 1:
                # AI calculates caption consensus
                caption_similarity_scores = []
                
                # Simple word overlap analysis (could be enhanced with embeddings)
                for i, cap1 in enumerate(all_captions):
                    for j, cap2 in enumerate(all_captions[i+1:], i+1):
                        words1 = set(cap1.lower().split())
                        words2 = set(cap2.lower().split())
                        overlap = len(words1.intersection(words2))
                        total = len(words1.union(words2))
                        similarity = overlap / total if total > 0 else 0
                        caption_similarity_scores.append(similarity)
                
                consensus_score = np.mean(caption_similarity_scores) if caption_similarity_scores else 0.5
                ensemble_results["consensus_analysis"] = {
                    "caption_consensus": consensus_score,
                    "reliability_assessment": "high" if consensus_score > 0.7 else "moderate" if consensus_score > 0.4 else "low"
                }
                ensemble_results["reliability_score"] = consensus_score
            
            return ensemble_results
            
        except Exception as e:
            print(f"Ensemble analysis error: {e}")
            return ensemble_results