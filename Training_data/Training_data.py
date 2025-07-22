#!/usr/bin/env python3
"""
Multi-Metric SDXL Prompt Optimizer
Fixed version with DiffusionDB support and improved TIFA fallbacks
"""

# Core imports
import torch
import torch.nn as nn
import numpy as np
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import time
import warnings
import os
import gc

# Setup logging FIRST - before any other imports that might use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger.info("🚀 Starting Multi-Metric SDXL Prompt Optimizer v2.0")

# Check CUDA availability
if torch.cuda.is_available():
    logger.info(f"🖥️ CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    logger.warning("⚠️ CUDA not available, using CPU")

# Essential ML imports with error handling
try:
    import clip
    CLIP_AVAILABLE = True
    logger.info("✅ CLIP loaded successfully")
except ImportError as e:
    logger.error(f"❌ CLIP not available: {e}")
    CLIP_AVAILABLE = False

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        T5Tokenizer, T5ForConditionalGeneration,
        AutoTokenizer, AutoModel, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers loaded successfully")
except ImportError as e:
    logger.error(f"❌ Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    from diffusers import StableDiffusionXLPipeline
    DIFFUSERS_AVAILABLE = True
    logger.info("✅ Diffusers loaded successfully")
except ImportError as e:
    logger.error(f"❌ Diffusers not available: {e}")
    DIFFUSERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✅ spaCy loaded successfully")
except ImportError as e:
    logger.error(f"❌ spaCy not available: {e}")
    SPACY_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
    logger.info("✅ Datasets loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Datasets not available: {e}")
    DATASETS_AVAILABLE = False

# Optional imports with graceful fallbacks
try:
    from tifascore import tifa_score_benchmark, VQAModel
    TIFA_AVAILABLE = True
    logger.info("✅ TIFA loaded successfully")
except ImportError:
    logger.warning("⚠️ TIFA not available - using advanced fallback evaluation")
    TIFA_AVAILABLE = False

@dataclass
class PromptMetrics:
    """Container for prompt evaluation metrics"""
    tifa_score: float
    clip_score: float
    blip2_score: float
    aesthetic_score: float
    combined_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'tifa_score': self.tifa_score,
            'clip_score': self.clip_score,
            'blip2_score': self.blip2_score,
            'aesthetic_score': self.aesthetic_score,
            'combined_score': self.combined_score
        }

class DependencyChecker:
    """Check and report missing dependencies"""
    
    @staticmethod
    def check_essential_dependencies():
        """Check if essential dependencies are available"""
        missing = []
        
        if not CLIP_AVAILABLE:
            missing.append("clip-by-openai")
        if not TRANSFORMERS_AVAILABLE:
            missing.append("transformers")
        if not DIFFUSERS_AVAILABLE:
            missing.append("diffusers")
        if not SPACY_AVAILABLE:
            missing.append("spacy")
            
        if missing:
            logger.error(f"❌ Missing essential dependencies: {', '.join(missing)}")
            logger.error("Install with: pip install " + " ".join(missing))
            return False
        
        logger.info("✅ All essential dependencies available")
        return True
    
    @staticmethod
    def check_optional_dependencies():
        """Report optional dependencies status"""
        optional_status = []
        
        if not DATASETS_AVAILABLE:
            optional_status.append("⚠️ 'datasets' not available - limited dataset loading")
        else:
            optional_status.append("✅ 'datasets' available")
            
        if not TIFA_AVAILABLE:
            optional_status.append("⚠️ 'tifascore' not available - using advanced fallback TIFA evaluation")
        else:
            optional_status.append("✅ 'tifascore' available")
        
        for status in optional_status:
            logger.info(status)

class AdvancedDatasetLoader:
    """Advanced dataset loader with fixed DiffusionDB support and better fallbacks"""
    
    def __init__(self):
        self.available_datasets = self._check_dataset_availability()
        self.fallback_prompts = self._initialize_fallback_prompts()
        
    def _initialize_fallback_prompts(self) -> List[str]:
        """Initialize high-quality fallback prompts"""
        return [
            # Artistic & Creative
            "a majestic lion resting under a baobab tree in the African savanna, golden hour lighting",
            "a cozy coffee shop interior with warm Edison bulb lighting and vintage leather furniture",
            "a futuristic cityscape at sunset with flying cars and holographic advertisements",
            "a magical forest with glowing mushrooms, fireflies, and ancient moss-covered trees",
            "a cyberpunk street scene with neon lights reflecting on wet pavement",
            
            # Nature & Landscapes
            "a serene mountain lake reflecting snow-capped peaks under a starry night sky",
            "a beautiful garden with colorful wildflowers and butterflies on a sunny day",
            "a peaceful beach scene with palm trees and crystal clear turquoise water",
            "a dramatic waterfall cascading through a lush tropical rainforest",
            "a desert oasis with palm trees and clear blue spring water",
            
            # Architecture & Interiors
            "a gothic cathedral with intricate stained glass windows and dramatic lighting",
            "a modern minimalist kitchen with marble countertops and natural lighting",
            "a rustic farmhouse surrounded by rolling green hills and wildflowers",
            "a Japanese zen garden with cherry blossoms and a traditional tea house",
            "a steampunk laboratory with brass instruments and glowing Edison bulbs",
            
            # Fantasy & Sci-Fi
            "a space station orbiting a distant nebula with colorful cosmic clouds",
            "a medieval castle on a hilltop during golden hour with dramatic clouds",
            "an underwater coral reef teeming with colorful tropical fish",
            "a vintage steam locomotive crossing a stone bridge in a mountain valley",
            "a modern art gallery with abstract paintings and clean white walls",
            
            # Portraits & Characters
            "a portrait of an elderly man with kind eyes, weathered hands, and gentle smile",
            "a young artist painting in a sunlit studio filled with canvases",
            "a chef preparing food in a busy restaurant kitchen with dramatic lighting",
            "a musician playing violin on a dimly lit concert stage",
            "a child reading a book under a large oak tree in summer",
        ]
    
    def _check_dataset_availability(self):
        """Check which datasets are accessible with detailed error reporting"""
        available = []
        
        if not DATASETS_AVAILABLE:
            logger.warning("📦 Datasets library not available - using fallback prompts only")
            return available
        
        # Test LAION Aesthetics
        try:
            logger.info("🔍 Testing LAION Aesthetics access...")
            test_dataset = load_dataset(
                "ChristophSchuhmann/improved_aesthetics_6.5plus", 
                split="train", 
                streaming=True
            )
            next(iter(test_dataset))  # Try to get one sample
            available.append("laion_aesthetics")
            logger.info("✅ LAION Aesthetics dataset accessible")
        except Exception as e:
            logger.warning(f"⚠️ LAION Aesthetics not accessible: {str(e)[:100]}...")
        
        # Test DiffusionDB with trust_remote_code=True (FIXED!)
        try:
            logger.info("🔍 Testing DiffusionDB access...")
            test_dataset = load_dataset(
                "poloclub/diffusiondb", 
                "2m_random_1k", 
                split="train",
                trust_remote_code=True  # FIX: Added this parameter
            )
            available.append("diffusiondb")
            logger.info("✅ DiffusionDB dataset accessible")
        except Exception as e:
            logger.warning(f"⚠️ DiffusionDB not accessible: {str(e)[:100]}...")
        
        # Test Stable Diffusion Prompts
        try:
            logger.info("🔍 Testing Stable Diffusion Prompts access...")
            test_dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
            available.append("stable_diffusion_prompts")
            logger.info("✅ Stable Diffusion Prompts dataset accessible")
        except Exception as e:
            logger.warning(f"⚠️ Stable Diffusion Prompts not accessible: {str(e)[:100]}...")
        
        if available:
            logger.info(f"📊 Available datasets: {', '.join(available)}")
        else:
            logger.warning("📦 No external datasets available - using fallback prompts")
        
        return available
    
    def load_available_datasets(self, max_samples: int = 1000) -> Tuple[List[str], List[float]]:
        """Load from available datasets with improved error handling"""
        all_prompts = []
        all_scores = []
        
        if not self.available_datasets:
            logger.info("📦 Using fallback prompts (no external datasets available)")
            return self._load_fallback_prompts()
        
        samples_per_dataset = max_samples // len(self.available_datasets)
        logger.info(f"📊 Loading ~{samples_per_dataset} samples from each available dataset")
        
        # Load LAION Aesthetics
        if "laion_aesthetics" in self.available_datasets:
            try:
                prompts, scores = self._load_laion_aesthetics(samples_per_dataset)
                all_prompts.extend(prompts)
                all_scores.extend(scores)
                logger.info(f"📊 Loaded {len(prompts)} LAION prompts (avg score: {np.mean(scores):.2f})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load LAION: {e}")
        
        # Load DiffusionDB (FIXED!)
        if "diffusiondb" in self.available_datasets:
            try:
                prompts = self._load_diffusiondb(samples_per_dataset)
                all_prompts.extend(prompts)
                all_scores.extend([6.5] * len(prompts))  # Default score for user-generated
                logger.info(f"📊 Loaded {len(prompts)} DiffusionDB prompts")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load DiffusionDB: {e}")
        
        # Load Stable Diffusion Prompts
        if "stable_diffusion_prompts" in self.available_datasets:
            try:
                prompts = self._load_stable_diffusion_prompts(samples_per_dataset)
                all_prompts.extend(prompts)
                all_scores.extend([7.0] * len(prompts))  # Higher score for curated
                logger.info(f"📊 Loaded {len(prompts)} curated SD prompts")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load SD prompts: {e}")
        
        # Add fallback prompts if we didn't get enough
        if len(all_prompts) < max_samples // 2:
            logger.info("📦 Adding fallback prompts to reach target count")
            fallback_prompts, fallback_scores = self._load_fallback_prompts()
            needed = max_samples - len(all_prompts)
            all_prompts.extend(fallback_prompts[:needed])
            all_scores.extend(fallback_scores[:needed])
        
        logger.info(f"📊 Total loaded: {len(all_prompts)} prompts (avg score: {np.mean(all_scores):.2f})")
        return all_prompts, all_scores
    
    def _load_fallback_prompts(self) -> Tuple[List[str], List[float]]:
        """Load high-quality fallback prompts"""
        logger.info("📦 Loading fallback prompt collection...")
        
        # Assign realistic quality scores based on prompt complexity
        scores = []
        for prompt in self.fallback_prompts:
            # Score based on prompt length and descriptive terms
            base_score = 7.0
            if len(prompt.split()) > 10:
                base_score += 0.5
            if any(term in prompt.lower() for term in ['dramatic', 'golden hour', 'professional', 'award winning']):
                base_score += 0.3
            if any(term in prompt.lower() for term in ['lighting', 'composition', 'detailed']):
                base_score += 0.2
            
            # Add some randomness
            final_score = min(base_score + random.uniform(-0.2, 0.3), 8.5)
            scores.append(final_score)
        
        logger.info(f"📦 Loaded {len(self.fallback_prompts)} fallback prompts (avg score: {np.mean(scores):.2f})")
        return self.fallback_prompts.copy(), scores
    
    def _load_laion_aesthetics(self, max_samples: int) -> Tuple[List[str], List[float]]:
        """Load LAION Aesthetics dataset"""
        dataset = load_dataset(
            "ChristophSchuhmann/improved_aesthetics_6.5plus", 
            split="train", 
            streaming=True
        )
        
        prompts = []
        scores = []
        
        logger.info(f"🔄 Loading LAION Aesthetics (target: {max_samples} samples)...")
        
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = sample.get('TEXT', '').strip()
            if prompt and len(prompt) > 10 and len(prompt) < 200:  # Quality filters
                prompts.append(prompt)
                scores.append(float(sample.get('aesthetic', 6.5)))
            
            if i % 100 == 0 and i > 0:
                logger.info(f"   📥 Processed {i} samples, kept {len(prompts)} quality prompts")
        
        return prompts, scores
    
    def _load_diffusiondb(self, max_samples: int) -> List[str]:
        """Load DiffusionDB prompts with trust_remote_code=True (FIXED!)"""
        dataset = load_dataset(
            "poloclub/diffusiondb", 
            "2m_random_1k", 
            split="train",
            trust_remote_code=True  # FIX: This was the missing parameter!
        )
        
        prompts = []
        logger.info(f"🔄 Loading DiffusionDB (target: {max_samples} samples)...")
        
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = sample.get('prompt', '').strip()
            if prompt and len(prompt) > 10 and len(prompt) < 200:  # Quality filters
                prompts.append(prompt)
            
            if i % 100 == 0 and i > 0:
                logger.info(f"   📥 Processed {i} samples, kept {len(prompts)} quality prompts")
        
        return prompts
    
    def _load_stable_diffusion_prompts(self, max_samples: int) -> List[str]:
        """Load curated Stable Diffusion prompts"""
        dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train")
        
        prompts = []
        logger.info(f"🔄 Loading SD Prompts (target: {max_samples} samples)...")
        
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            prompt = sample.get('text', '').strip()
            if prompt and len(prompt) > 10 and len(prompt) < 200:  # Quality filters
                prompts.append(prompt)
        
        return prompts

class AdvancedMetricEvaluator:
    """Advanced metric evaluator with improved TIFA fallbacks and better error handling"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"🖥️ Using device: {self.device}")
        
        self.models_loaded = self._load_models()
        self._print_model_status()
        
    def _print_model_status(self):
        """Print status of all loaded models"""
        logger.info("📋 Model Status:")
        for model_name, status in self.models_loaded.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {model_name.upper()}: {'Available' if status else 'Not Available'}")
    
    def _load_models(self):
        """Load available models with graceful fallbacks and memory management"""
        loaded = {}
        
        # Load CLIP with better error handling
        if CLIP_AVAILABLE:
            try:
                logger.info("🔄 Loading CLIP model...")
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                loaded['clip'] = True
                logger.info("✅ CLIP model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ CLIP model failed to load: {e}")
                loaded['clip'] = False
        else:
            loaded['clip'] = False
        
        # Load BLIP2 with memory management
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("🔄 Loading BLIP2 model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self.device)
                
                # Enable memory efficient attention if available
                if hasattr(self.blip_model, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.blip_model.enable_xformers_memory_efficient_attention()
                    except:
                        pass
                
                loaded['blip2'] = True
                logger.info("✅ BLIP2 model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ BLIP2 model failed to load: {e}")
                loaded['blip2'] = False
        else:
            loaded['blip2'] = False
        
        # Load spaCy with multiple fallback options
        if SPACY_AVAILABLE:
            spacy_models = ["en_core_web_sm", "en", "en_core_web_md"]
            for model_name in spacy_models:
                try:
                    logger.info(f"🔄 Loading spaCy model: {model_name}")
                    self.nlp = spacy.load(model_name)
                    loaded['spacy'] = True
                    logger.info(f"✅ spaCy model '{model_name}' loaded successfully")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ spaCy model '{model_name}' failed: {e}")
                    continue
            else:
                loaded['spacy'] = False
                logger.warning("⚠️ No spaCy models could be loaded")
        else:
            loaded['spacy'] = False
        
        # Load TIFA if available
        if TIFA_AVAILABLE:
            try:
                logger.info("🔄 Loading TIFA VQA model...")
                self.tifa_vqa = VQAModel("mplug-large")
                loaded['tifa'] = True
                logger.info("✅ TIFA VQA model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ TIFA VQA model failed to load: {e}")
                loaded['tifa'] = False
        else:
            loaded['tifa'] = False
        
        return loaded
    
    def evaluate_clip_score(self, image, prompt: str) -> float:
        """Evaluate CLIP score with robust error handling"""
        if not self.models_loaded.get('clip', False):
            logger.debug("CLIP not available, returning baseline score")
            return 0.5
        
        try:
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([prompt], truncate=True).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(image_features, text_features)
                return float(similarity.item())
        
        except Exception as e:
            logger.warning(f"CLIP evaluation failed: {e}")
            return 0.5
    
    def evaluate_tifa_score(self, image, prompt: str) -> float:
        """Evaluate TIFA score with advanced fallback"""
        if self.models_loaded.get('tifa', False):
            try:
                # Use actual TIFA evaluation if available
                score = self.tifa_vqa.evaluate(image, prompt)
                return float(score)
            except Exception as e:
                logger.warning(f"TIFA evaluation failed: {e}")
        
        # Advanced fallback TIFA evaluation
        return self._advanced_tifa_fallback(image, prompt)
    
    def _advanced_tifa_fallback(self, image, prompt: str) -> float:
        """Advanced TIFA fallback using multiple evaluation strategies"""
        scores = []
        
        # Strategy 1: Image captioning + text similarity
        if self.models_loaded.get('blip2', False):
            try:
                caption_score = self._caption_similarity_score(image, prompt)
                scores.append(caption_score)
            except Exception as e:
                logger.debug(f"Caption similarity failed: {e}")
        
        # Strategy 2: Entity and keyword matching
        if self.models_loaded.get('spacy', False):
            try:
                entity_score = self._entity_matching_score(image, prompt)
                scores.append(entity_score)
            except Exception as e:
                logger.debug(f"Entity matching failed: {e}")
        
        # Strategy 3: CLIP-based content verification
        if self.models_loaded.get('clip', False):
            try:
                content_score = self._clip_content_verification(image, prompt)
                scores.append(content_score)
            except Exception as e:
                logger.debug(f"Content verification failed: {e}")
        
        # Return weighted average or default
        if scores:
            return np.mean(scores)
        else:
            logger.debug("All TIFA fallback strategies failed, using default score")
            return 0.6
    
    def _caption_similarity_score(self, image, prompt: str) -> float:
        """Compare prompt with generated image caption"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Use CLIP similarity if available, otherwise text similarity
        if self.models_loaded.get('clip', False):
            return self.evaluate_clip_score(image, caption)
        else:
            return self._simple_text_similarity(prompt, caption)
    
    def _entity_matching_score(self, image, prompt: str) -> float:
        """Score based on entity and keyword matching"""
        # Generate caption first
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        # Analyze with spaCy
        prompt_doc = self.nlp(prompt)
        caption_doc = self.nlp(caption)
        
        # Extract entities
        prompt_entities = {ent.text.lower() for ent in prompt_doc.ents}
        caption_entities = {ent.text.lower() for ent in caption_doc.ents}
        
        # Extract important tokens
        prompt_tokens = {token.lemma_.lower() for token in prompt_doc 
                        if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and len(token.text) > 2}
        caption_tokens = {token.lemma_.lower() for token in caption_doc 
                         if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and len(token.text) > 2}
        
        # Calculate scores
        entity_overlap = len(prompt_entities.intersection(caption_entities))
        token_overlap = len(prompt_tokens.intersection(caption_tokens))
        
        entity_score = entity_overlap / max(len(prompt_entities), 1)
        token_score = token_overlap / max(len(prompt_tokens), 1)
        
        # Weighted combination
        return (entity_score * 0.6 + token_score * 0.4)
    
    def _clip_content_verification(self, image, prompt: str) -> float:
        """Use CLIP to verify specific content elements"""
        # Extract key elements from prompt
        doc = self.nlp(prompt) if self.models_loaded.get('spacy', False) else None
        
        if doc:
            # Test specific elements
            elements = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop]
            element_scores = []
            
            for element in elements[:5]:  # Test top 5 elements
                element_prompt = f"a {element}"
                score = self.evaluate_clip_score(image, element_prompt)
                element_scores.append(score)
            
            return np.mean(element_scores) if element_scores else 0.6
        else:
            # Fallback: just use the original prompt
            return self.evaluate_clip_score(image, prompt)
    
    def evaluate_aesthetic_score(self, image) -> float:
        """Evaluate aesthetic score using multiple criteria"""
        if not self.models_loaded.get('clip', False):
            return 0.7
        
        aesthetic_criteria = [
            # Quality terms
            "beautiful artwork", "professional photography", "high quality art",
            "masterpiece", "award winning", "stunning visuals",
            
            # Composition terms
            "perfect composition", "beautiful lighting", "artistic composition",
            "professional quality", "visually appealing"
        ]
        
        scores = []
        for criterion in aesthetic_criteria:
            try:
                score = self.evaluate_clip_score(image, criterion)
                scores.append(score)
            except:
                continue
        
        if scores:
            # Use weighted average - emphasize top scores
            sorted_scores = sorted(scores, reverse=True)
            weights = [0.3, 0.25, 0.2, 0.15, 0.1] + [0.0] * (len(sorted_scores) - 5)
            weighted_score = sum(s * w for s, w in zip(sorted_scores, weights))
            return weighted_score
        else:
            return 0.7
    
    def evaluate_blip2_score(self, image, prompt: str) -> float:
        """Evaluate BLIP2 score with improved logic"""
        if not self.models_loaded.get('blip2', False):
            return 0.6
        
        try:
            # Generate caption
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Score the caption quality and relevance
            if self.models_loaded.get('clip', False):
                # Use CLIP to compare original prompt with generated caption
                caption_score = self.evaluate_clip_score(image, caption)
                prompt_score = self.evaluate_clip_score(image, prompt)
                
                # Combine scores - good caption should be relevant to both image and prompt
                return (caption_score + prompt_score) / 2
            else:
                # Simple text similarity fallback
                return self._simple_text_similarity(prompt, caption)
        
        except Exception as e:
            logger.warning(f"BLIP2 evaluation failed: {e}")
            return 0.6
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Enhanced text similarity using word overlap and position"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Bonus for important word matches
        important_words = {'beautiful', 'professional', 'detailed', 'high', 'quality', 'art'}
        important_matches = len(intersection.intersection(important_words))
        bonus = min(important_matches * 0.1, 0.3)
        
        return min(jaccard + bonus, 1.0)
    
    def evaluate_all_metrics(self, image, prompt: str, weights: Optional[Dict[str, float]] = None) -> PromptMetrics:
        """Evaluate all available metrics with progress tracking"""
        if weights is None:
            weights = {'tifa': 0.3, 'clip': 0.3, 'blip2': 0.2, 'aesthetic': 0.2}
        
        logger.debug(f"📊 Evaluating metrics for: {prompt[:50]}...")
        
        # Evaluate individual metrics
        tifa_score = self.evaluate_tifa_score(image, prompt)
        clip_score = self.evaluate_clip_score(image, prompt)
        blip2_score = self.evaluate_blip2_score(image, prompt)
        aesthetic_score = self.evaluate_aesthetic_score(image)
        
        # Calculate combined score
        combined_score = (
            weights['tifa'] * tifa_score +
            weights['clip'] * clip_score +
            weights['blip2'] * blip2_score +
            weights['aesthetic'] * aesthetic_score
        )
        
        return PromptMetrics(
            tifa_score=tifa_score,
            clip_score=clip_score,
            blip2_score=blip2_score,
            aesthetic_score=aesthetic_score,
            combined_score=combined_score
        )

class AdvancedPromptEnhancer:
    """Advanced prompt enhancement with multiple strategies"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.load_models()
    
    def load_models(self):
        """Load enhancement models with better error handling"""
        self.superprompt_available = False
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("🔄 Loading SuperPrompt model...")
                self.superprompt_model = T5ForConditionalGeneration.from_pretrained(
                    "roborovski/superprompt-v1"
                ).to(self.device)
                self.superprompt_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
                self.superprompt_available = True
                logger.info("✅ SuperPrompt model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ SuperPrompt not available: {e}")
                self.superprompt_available = False
        
        # Initialize enhancement templates
        self.enhancement_templates = {
            'quality': [
                "highly detailed, professional photography",
                "award winning art, masterpiece quality",
                "photorealistic, 8k resolution, sharp focus",
                "beautiful composition, perfect lighting",
                "cinematic lighting, dramatic atmosphere"
            ],
            'style': [
                "oil painting, classical art style",
                "digital art, concept art style",
                "photography, studio lighting",
                "watercolor painting, soft colors",
                "pencil drawing, detailed sketching"
            ],
            'mood': [
                "warm and inviting atmosphere",
                "dramatic and moody lighting",
                "bright and cheerful colors",
                "mysterious and atmospheric",
                "peaceful and serene mood"
            ]
        }
    
    def enhance_prompt(self, prompt: str) -> List[str]:
        """Generate enhanced prompt variations using multiple strategies"""
        variations = [prompt]  # Always include original
        
        # Strategy 1: SuperPrompt enhancement
        if self.superprompt_available:
            try:
                enhanced = self._superprompt_enhance(prompt)
                if enhanced and enhanced != prompt and len(enhanced) > len(prompt):
                    variations.append(enhanced)
                    logger.debug(f"SuperPrompt: {enhanced[:50]}...")
            except Exception as e:
                logger.warning(f"SuperPrompt enhancement failed: {e}")
        
        # Strategy 2: Template-based enhancements
        template_variations = self._template_enhancements(prompt)
        variations.extend(template_variations)
        
        # Strategy 3: Smart keyword additions
        keyword_variations = self._smart_keyword_enhancements(prompt)
        variations.extend(keyword_variations)
        
        # Remove duplicates and limit count
        unique_variations = self._deduplicate_variations(variations)
        
        logger.debug(f"Generated {len(unique_variations)} unique variations")
        return unique_variations[:8]  # Limit to 8 variations
    
    def _superprompt_enhance(self, prompt: str) -> str:
        """Enhance using SuperPrompt model with better error handling"""
        input_text = f"Expand the following prompt to add more detail: {prompt}"
        
        try:
            inputs = self.superprompt_tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=77, 
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.superprompt_model.generate(
                    inputs.input_ids,
                    max_new_tokens=77,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.superprompt_tokenizer.eos_token_id
                )
            
            enhanced = self.superprompt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output
            if input_text in enhanced:
                enhanced = enhanced.replace(input_text, "").strip()
            
            return enhanced if len(enhanced) > len(prompt) else prompt
            
        except Exception as e:
            logger.warning(f"SuperPrompt generation failed: {e}")
            return prompt
    
    def _template_enhancements(self, prompt: str) -> List[str]:
        """Generate template-based enhancements"""
        enhancements = []
        
        # Add quality improvements
        for quality_term in self.enhancement_templates['quality'][:3]:
            enhanced = f"{prompt}, {quality_term}"
            enhancements.append(enhanced)
        
        # Add style variations (only if prompt doesn't already specify style)
        if not any(style_word in prompt.lower() for style_word in ['painting', 'photography', 'art', 'drawing']):
            for style_term in self.enhancement_templates['style'][:2]:
                enhanced = f"{prompt}, {style_term}"
                enhancements.append(enhanced)
        
        return enhancements
    
    def _smart_keyword_enhancements(self, prompt: str) -> List[str]:
        """Add smart keyword enhancements based on prompt analysis"""
        enhancements = []
        prompt_lower = prompt.lower()
        
        # Analyze prompt content and add relevant keywords
        if any(word in prompt_lower for word in ['portrait', 'person', 'man', 'woman', 'face']):
            enhancements.append(f"{prompt}, detailed facial features, expressive eyes")
        
        if any(word in prompt_lower for word in ['landscape', 'nature', 'mountain', 'forest', 'ocean']):
            enhancements.append(f"{prompt}, natural lighting, wide angle view")
        
        if any(word in prompt_lower for word in ['city', 'building', 'architecture']):
            enhancements.append(f"{prompt}, urban photography, architectural details")
        
        if any(word in prompt_lower for word in ['animal', 'cat', 'dog', 'bird', 'lion']):
            enhancements.append(f"{prompt}, wildlife photography, natural habitat")
        
        # Add atmospheric enhancements
        if 'night' not in prompt_lower and 'dark' not in prompt_lower:
            enhancements.append(f"{prompt}, golden hour lighting")
        
        return enhancements[:3]  # Limit to 3 smart enhancements
    
    def _deduplicate_variations(self, variations: List[str]) -> List[str]:
        """Remove duplicates while preserving order and quality"""
        seen = set()
        unique_variations = []
        
        for variation in variations:
            # Normalize for comparison
            normalized = variation.lower().strip()
            if normalized not in seen and len(variation.strip()) > 0:
                seen.add(normalized)
                unique_variations.append(variation.strip())
        
        return unique_variations

class PromptOptimizer:
    """Main prompt optimization engine with advanced features"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Check dependencies
        if not DependencyChecker.check_essential_dependencies():
            raise RuntimeError("Essential dependencies missing - please install required packages")
        
        DependencyChecker.check_optional_dependencies()
        
        # Initialize components
        logger.info("🔧 Initializing optimization components...")
        self.dataset_loader = AdvancedDatasetLoader()
        self.evaluator = AdvancedMetricEvaluator(self.device)
        self.enhancer = AdvancedPromptEnhancer(self.device)
        
        # Load SDXL pipeline
        self.load_sdxl_pipeline()
        
        logger.info("🎉 Prompt Optimizer initialized successfully!")
    
    def load_sdxl_pipeline(self):
        """Load SDXL pipeline with comprehensive error handling"""
        if not DIFFUSERS_AVAILABLE:
            logger.error("❌ Diffusers not available - cannot load SDXL pipeline")
            self.sdxl_pipe = None
            return
        
        try:
            logger.info("🎨 Loading SDXL pipeline (this may take a few minutes)...")
            
            # Load with appropriate precision
            torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
            
            self.sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.device == 'cuda' else None
            ).to(self.device)
            
            # Enable optimizations
            if self.device == 'cuda':
                # Enable memory efficient attention if available
                if hasattr(self.sdxl_pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.sdxl_pipe.enable_xformers_memory_efficient_attention()
                        logger.info("✅ Memory efficient attention enabled")
                    except:
                        logger.info("⚠️ Memory efficient attention not available")
                
                # Enable attention slicing for lower memory usage
                self.sdxl_pipe.enable_attention_slicing()
                
                # Enable CPU offload if needed
                if torch.cuda.get_device_properties(0).total_memory < 12e9:  # Less than 12GB
                    self.sdxl_pipe.enable_model_cpu_offload()
                    logger.info("✅ CPU offload enabled for lower memory usage")
            
            logger.info("✅ SDXL pipeline loaded and optimized successfully")
        
        except Exception as e:
            logger.error(f"❌ Failed to load SDXL pipeline: {e}")
            self.sdxl_pipe = None
    
    def optimize_prompt(self, prompt: str, target_score: float = 0.75, max_iterations: int = 8) -> Tuple[str, PromptMetrics]:
        """Optimize a single prompt with advanced strategies"""
        if not self.sdxl_pipe:
            logger.error("❌ SDXL pipeline not available - cannot optimize prompts")
            return prompt, None
        
        logger.info(f"🎯 Optimizing prompt: '{prompt[:50]}...'")
        start_time = time.time()
        
        # Generate variations
        variations = self.enhancer.enhance_prompt(prompt)
        logger.info(f"📝 Generated {len(variations)} variations to test")
        
        best_prompt = prompt
        best_metrics = None
        best_score = 0.0
        
        for i, variant in enumerate(variations):
            logger.info(f"🔄 Testing variation {i+1}/{len(variations)}")
            logger.info(f"   📝 Prompt: '{variant[:60]}...'")
            
            try:
                # Generate image with error handling
                generation_start = time.time()
                
                # Clear cache before generation
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                image = self.sdxl_pipe(
                    variant,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=1024,
                    width=1024
                ).images[0]
                
                generation_time = time.time() - generation_start
                logger.info(f"   ⏱️ Generation time: {generation_time:.1f}s")
                
                # Evaluate metrics
                metrics = self.evaluator.evaluate_all_metrics(image, variant)
                
                logger.info(f"   📊 Combined Score: {metrics.combined_score:.3f}")
                logger.info(f"   📊 Breakdown - TIFA: {metrics.tifa_score:.3f}, "
                          f"CLIP: {metrics.clip_score:.3f}, "
                          f"BLIP2: {metrics.blip2_score:.3f}, "
                          f"Aesthetic: {metrics.aesthetic_score:.3f}")
                
                # Update best if this is better
                if metrics.combined_score > best_score:
                    best_score = metrics.combined_score
                    best_prompt = variant
                    best_metrics = metrics
                    logger.info(f"   🌟 New best score: {best_score:.3f}")
                
                # Early stopping if target reached
                if best_score >= target_score:
                    logger.info(f"   🎯 Target score {target_score:.3f} reached! Stopping early.")
                    break
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to evaluate variation: {e}")
                continue
        
        total_time = time.time() - start_time
        improvement = best_score - 0.5  # Baseline score
        
        logger.info(f"✅ Optimization complete!")
        logger.info(f"   ⏱️ Total time: {total_time:.1f}s")
        logger.info(f"   🏆 Best score: {best_score:.3f}")
        logger.info(f"   📈 Improvement: {improvement:.3f}")
        
        return best_prompt, best_metrics

def main():
    """Main function with comprehensive testing"""
    try:
        logger.info("🚀 Initializing Multi-Metric SDXL Prompt Optimizer...")
        
        # Initialize optimizer
        optimizer = PromptOptimizer()
        
        # Test prompts
        test_prompts = [
            "a dog wearing a hat",
            "a beautiful landscape",
            "a futuristic city"
        ]
        
        logger.info(f"🧪 Testing with {len(test_prompts)} prompts...")
        
        results = []
        for i, test_prompt in enumerate(test_prompts):
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 Test {i+1}/{len(test_prompts)}: '{test_prompt}'")
            logger.info(f"{'='*50}")
            
            optimized_prompt, metrics = optimizer.optimize_prompt(test_prompt, target_score=0.7)
            
            result = {
                'original': test_prompt,
                'optimized': optimized_prompt,
                'metrics': metrics.to_dict() if metrics else None
            }
            results.append(result)
        
        # Print final results
        print("\n" + "="*80)
        print("🎉 FINAL OPTIMIZATION RESULTS")
        print("="*80)
        
        for i, result in enumerate(results):
            print(f"\n📝 Test {i+1}:")
            print(f"   Original:  {result['original']}")
            print(f"   Optimized: {result['optimized']}")
            
            if result['metrics']:
                m = result['metrics']
                print(f"   📊 Scores: Combined={m['combined_score']:.3f}, "
                      f"TIFA={m['tifa_score']:.3f}, "
                      f"CLIP={m['clip_score']:.3f}, "
                      f"BLIP2={m['blip2_score']:.3f}, "
                      f"Aesthetic={m['aesthetic_score']:.3f}")
        
        print("="*80)
        logger.info("🎉 All tests completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Optimization interrupted by user")
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()