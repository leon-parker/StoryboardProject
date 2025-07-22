from sentence_transformers import SentenceTransformer
import spacy
# PureAITIFAMetric - Extracted directly from original code

import config
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

class PureAITIFAMetric:
    """
    Text-Image Faithfulness Assessment using PURE AI
    NO hard-coded keywords - AI models extract all semantic components
    """
    
    def __init__(self):
        print("🎯 Loading PURE AI TIFA (Text-Image Faithfulness) metric...")
        
        try:
            # Enhanced CLIP for visual-semantic alignment
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            
            # Load NLP model for semantic parsing (NO KEYWORDS)
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_available = True
                print("✅ SpaCy NLP loaded for semantic parsing")
            except:
                print("⚠️ SpaCy not available - using basic parsing")
                self.spacy_available = False
            
            # Sentence transformer for semantic embeddings
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.to("cuda")
                
            self.available = True
            print("✅ PURE AI TIFA metric loaded successfully")
            
        except Exception as e:
            print(f"⚠️ TIFA unavailable: {e}")
            self.available = False
    
    def ai_extract_semantic_components(self, text):
        """
        PURE AI semantic extraction - NO hard-coded keywords
        Uses NLP models to automatically detect semantic components
        """
        components = {
            'entities': [],      # AI-detected entities
            'actions': [],       # AI-detected actions/verbs
            'attributes': [],    # AI-detected adjectives/descriptions
            'spatial_relations': [],  # AI-detected spatial relationships
            'concepts': []       # AI-detected abstract concepts
        }
        
        if self.spacy_available:
            # Use SpaCy for advanced semantic parsing
            doc = self.nlp(text)
            
            # AI extracts entities automatically
            for ent in doc.ents:
                components['entities'].append({
                    'text': ent.text.lower(),
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_
                })
            
            # AI extracts actions/verbs automatically  
            for token in doc:
                if token.pos_ == "VERB" and not token.is_stop:
                    components['actions'].append({
                        'text': token.lemma_.lower(),
                        'tense': token.tag_,
                        'dependency': token.dep_
                    })
                
                # AI extracts attributes automatically
                elif token.pos_ == "ADJ" and not token.is_stop:
                    components['attributes'].append({
                        'text': token.lemma_.lower(),
                        'intensity': 'moderate'  # Could be enhanced with sentiment analysis
                    })
                
                # AI extracts spatial relations automatically
                elif token.dep_ in ["prep", "pobj", "advmod"] and token.pos_ in ["ADP", "ADV"]:
                    components['spatial_relations'].append({
                        'text': token.text.lower(),
                        'dependency': token.dep_
                    })
            
            # AI extracts noun concepts automatically
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Avoid overly long phrases
                    components['concepts'].append({
                        'text': chunk.text.lower(),
                        'root': chunk.root.lemma_.lower()
                    })
        
        else:
            # Fallback: basic AI-driven parsing without SpaCy
            words = text.lower().split()
            # Simple POS detection using basic rules - still no hard-coded keywords
            for word in words:
                # Basic heuristic detection (could be replaced with more AI models)
                if word.endswith('ing') or word.endswith('ed'):
                    components['actions'].append({'text': word, 'type': 'detected_verb'})
                elif word.endswith('ly'):
                    components['attributes'].append({'text': word, 'type': 'detected_adverb'})
                else:
                    components['concepts'].append({'text': word, 'type': 'detected_concept'})
        
        return components
    
    def calculate_semantic_overlap(self, prompt_components, caption_components):
        """
        Calculate semantic overlap using AI embeddings - NO keyword matching
        """
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
                
                # Calculate maximum semantic similarity for each prompt component
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
    
    def calculate_pure_ai_tifa_score(self, image, prompt, generated_caption):
        """Calculate TIFA score using PURE AI analysis - NO keywords"""
        if not self.available:
            return {"score": 0.5, "details": "TIFA unavailable"}
        
        try:
            # AI extracts semantic components from both prompt and caption
            prompt_components = self.ai_extract_semantic_components(prompt)
            caption_components = self.ai_extract_semantic_components(generated_caption)
            
            # AI calculates semantic overlap using embeddings
            semantic_overlaps = self.calculate_semantic_overlap(prompt_components, caption_components)
            
            # CLIP semantic similarity analysis
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
                
                # AI calculates multi-dimensional similarities
                img_prompt_sim = torch.cosine_similarity(
                    outputs.image_embeds, 
                    outputs.text_embeds[0:1], 
                    dim=1
                ).item()
                
                img_caption_sim = torch.cosine_similarity(
                    outputs.image_embeds, 
                    outputs.text_embeds[1:2], 
                    dim=1
                ).item()
                
                prompt_caption_sim = torch.cosine_similarity(
                    outputs.text_embeds[0:1], 
                    outputs.text_embeds[1:2], 
                    dim=1
                ).item()
            
            # AI-driven final score calculation
            clip_score = (img_prompt_sim + prompt_caption_sim) / 2
            semantic_score = np.mean(list(semantic_overlaps.values())) if semantic_overlaps else 0.5
            
            # Weighted combination favoring CLIP (more reliable)
            tifa_score = (clip_score * 0.8) + (semantic_score * 0.2)
            
            return {
                "score": float(tifa_score),
                "details": {
                    "clip_semantic_score": float(clip_score),
                    "ai_component_overlap": float(semantic_score),
                    "component_overlaps": semantic_overlaps,
                    "clip_similarities": {
                        "image_prompt": float(img_prompt_sim),
                        "image_caption": float(img_caption_sim),
                        "prompt_caption": float(prompt_caption_sim)
                    },
                    "ai_extracted_components": {
                        "prompt": prompt_components,
                        "caption": caption_components
                    }
                }
            }
            
        except Exception as e:
            print(f"Pure AI TIFA calculation error: {e}")
            return {"score": 0.5, "details": f"Error: {e}"}