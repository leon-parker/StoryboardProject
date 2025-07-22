import torch
from collections import defaultdict
import hashlib
import numpy as np
import config
import copy
from PIL import Image
import time
import uuid

# Initialize REAL HuggingFace CLIP (now working with torch 2.5.1)
try:
    from transformers import CLIPProcessor, CLIPModel
    print("🔍 Loading REAL HuggingFace CLIP for character consistency...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = clip_model.to(device)
    CLIP_AVAILABLE = True
    print("✅ REAL CLIP loaded successfully for consistency")
except Exception as e:
    print(f"❌ CLIP unavailable: {e}")
    CLIP_AVAILABLE = False
    clip_processor = None
    clip_model = None

class ConsistentCharacterManager:
    def __init__(self):
        self.clip_available = CLIP_AVAILABLE
        
    def analyze_character_consistency(self, image, prompt):
        if not self.clip_available:
            return {'consistency_score': 0.5, 'available': False}
            
        try:
            inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                similarity = outputs.logits_per_image[0][0].item()
                # Normalize to reasonable range
                score = max(0.3, min(1.0, similarity * 0.1 + 0.7))
                
            return {
                'consistency_score': score,
                'available': True,
                'message': f'REAL CLIP character consistency: {score:.3f}'
            }
        except Exception as e:
            return {'consistency_score': 0.5, 'available': False, 'message': f'Error: {e}'}

class StoryDiffusionConsistencyManager(ConsistentCharacterManager):
    """Enhanced consistency manager with additional StoryDiffusion features"""
    
    def __init__(self):
        super().__init__()
        self.character_embeddings = {}
        self.style_embeddings = {}
        
    def extract_character_embedding(self, image, character_name, store=True):
        """Extract character embedding, optionally store it"""
        if not self.clip_available:
            return None
            
        try:
            # Generate character-specific prompt for embedding
            char_prompt = f"character {character_name}, consistent appearance, same person"
            
            inputs = clip_processor(text=[char_prompt], images=image, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                image_embedding = outputs.image_embeds[0].cpu().numpy()
                
            # Only store if requested (for confirmed selections)
            if store:
                if character_name not in self.character_embeddings:
                    self.character_embeddings[character_name] = []
                
                self.character_embeddings[character_name].append(image_embedding)
                print(f"🎭 Stored embedding for character: {character_name}")
            
            return image_embedding
            
        except Exception as e:
            print(f"⚠️ Could not extract character embedding: {e}")
            return None
    
    def check_character_consistency(self, image, character_name, threshold=0.75):
        """Check if image is consistent with stored character embeddings"""
        if character_name not in self.character_embeddings or not self.clip_available:
            return {'consistent': True, 'score': 0.5, 'message': 'No reference available'}
        
        try:
            # Extract embedding for current image (don't store)
            current_embedding = self.extract_character_embedding(image, f"temp_{character_name}", store=False)
            if current_embedding is None:
                return {'consistent': True, 'score': 0.5, 'message': 'Could not extract embedding'}
            
            # Compare with stored embeddings
            stored_embeddings = self.character_embeddings[character_name]
            similarities = []
            
            for stored_embedding in stored_embeddings:
                # Calculate cosine similarity
                similarity = np.dot(current_embedding, stored_embedding) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append(similarity)
            
            # Use best similarity score
            max_similarity = max(similarities) if similarities else 0
            is_consistent = max_similarity >= threshold
            
            return {
                'consistent': is_consistent,
                'score': max_similarity,
                'threshold': threshold,
                'message': f'Character consistency: {max_similarity:.3f} (threshold: {threshold})'
            }
            
        except Exception as e:
            return {'consistent': True, 'score': 0.5, 'message': f'Error checking consistency: {e}'}

class ConsistencyEnhancedValidator:
    def __init__(self, parent=None):
        """
        Initialize ConsistencyEnhancedValidator with access to parent pipeline resources
        """
        self.validator_ready = True
        self.parent = parent
        
        # Access shared resources from parent
        if parent:
            self.pipe = parent.pipe
            self.validator = parent.validator
            self.ai_refiner = parent.ai_refiner
            self.device = "cuda" if hasattr(parent.pipe, 'device') else "cpu"
            
            print("✅ ConsistencyEnhancedValidator initialized with parent resources")
            print(f"   📱 Device: {self.device}")
            print(f"   🎨 Pipeline: Available")
            print(f"   🔍 Validator: Available") 
            print(f"   🔧 AI Refiner: {'Available' if parent.ai_refiner.available else 'Not Available'}")
        else:
            self.pipe = None
            self.validator = None
            self.ai_refiner = None
            self.device = "cpu"
            print("⚠️ ConsistencyEnhancedValidator initialized without parent - limited functionality")
        
        # Character consistency tracking
        self.character_embeddings = {}
        self.story_sessions = {}
        self.current_story_id = None
        
        # CRITICAL: Track pending image batches that haven't been confirmed yet
        self.pending_batches = {}  # batch_id -> batch_data
        
        # Initialize enhanced consistency manager
        self.consistency_manager = StoryDiffusionConsistencyManager()
        
    def validate(self, content):
        """Enhanced validation using parent resources"""
        if self.validator:
            return self.validator.enhanced_validate_image(content, "", "consistency_check")
        return True
    
    def start_consistent_story(self, story_name, style_description="calm cartoon autism-friendly style"):
        """Start a new consistent story session"""
        story_id = f"{story_name}_{int(time.time())}"
        
        self.story_sessions[story_id] = {
            'name': story_name,
            'style': style_description,
            'characters': {},
            'scenes': [],
            'created_at': time.time(),
            'consistency_scores': [],
            'character_consistency_scores': []
        }
        
        self.current_story_id = story_id
        print(f"📖 Started story session: {story_name} (ID: {story_id})")
        print(f"🎨 Style: {style_description}")
        
        return story_id
    
    def generate_consistent_scene_batch(self, scene_prompt, character_name="main_character", 
                                      story_id=None, num_candidates=3, consistency_strength=0.8, 
                                      **kwargs):
        """
        Generate multiple candidate images WITHOUT updating character memory
        
        Args:
            scene_prompt: Description of the scene
            character_name: Name of the character to maintain consistency for
            story_id: Story session ID (uses current if None)
            num_candidates: Number of candidate images to generate
            consistency_strength: How strongly to enforce consistency (0.0-1.0)
            **kwargs: Additional generation parameters
            
        Returns:
            dict: Batch results with all candidates and metadata (NO MEMORY UPDATES)
        """
        if not self.pipe:
            print("❌ No pipeline available - parent not provided")
            return None
            
        # Use current story or specified story
        story_id = story_id or self.current_story_id
        if not story_id or story_id not in self.story_sessions:
            print("⚠️ No active story session - starting default session")
            story_id = self.start_consistent_story("default_story")
        
        story_session = self.story_sessions[story_id]
        
        # Build consistency-enhanced prompt using EXISTING memory only
        style_prompt = story_session['style']
        enhanced_prompt = f"{scene_prompt}, {style_prompt}"
        
        # Add character consistency from CONFIRMED memory (not from this batch)
        character_consistency_features = []
        if character_name in story_session['characters']:
            char_features = story_session['characters'][character_name]
            consistency_features = char_features.get('features', [])
            if consistency_features:
                character_consistency_features = consistency_features[-3:]
                enhanced_prompt = f"{enhanced_prompt}, {', '.join(character_consistency_features)}"
                print(f"🎭 Applied character consistency from memory: {', '.join(character_consistency_features)}")
        
        print(f"🎬 Generating {num_candidates} candidate scenes: {scene_prompt}")
        print(f"📝 Enhanced prompt: {enhanced_prompt}")
        
        # Build negative prompt
        consistency_negative = [
            "inconsistent style", "different character", "style change", 
            "character inconsistency", "different person"
        ]
        
        if character_name in story_session['characters']:
            consistency_negative.extend([
                f"different {character_name}", "changed appearance", "different facial features"
            ])
        
        base_negative = kwargs.get('negative_prompt', "blurry, low quality, distorted, deformed")
        full_negative = f"{base_negative}, {', '.join(consistency_negative)}"
        
        # Generation settings
        generation_settings = {
            'prompt': enhanced_prompt,
            'negative_prompt': full_negative,
            'height': kwargs.get('height', 1024),
            'width': kwargs.get('width', 1024),
            'num_inference_steps': kwargs.get('num_inference_steps', 25),
            'guidance_scale': kwargs.get('guidance_scale', 7.0 + consistency_strength),
            'generator': kwargs.get('generator')
        }
        
        try:
            start_time = time.time()
            
            # Generate multiple candidates
            candidates = []
            candidate_validations = []
            candidate_consistency_checks = []
            
            print(f"🔄 Generating {num_candidates} candidates...")
            for i in range(num_candidates):
                print(f"   Generating candidate {i+1}/{num_candidates}...")
                
                result = self.pipe(**generation_settings)
                candidate_image = result.images[0]
                
                # Validate each candidate (but don't update memory)
                if self.validator:
                    validation = self.validator.enhanced_validate_image(
                        candidate_image, enhanced_prompt, f"candidate_{i}"
                    )
                else:
                    validation = {'quality_score': 0.5, 'message': 'No validator available'}
                
                # Check character consistency (but don't store embedding)
                character_consistency = self.consistency_manager.check_character_consistency(
                    candidate_image, character_name
                )
                
                candidates.append(candidate_image)
                candidate_validations.append(validation)
                candidate_consistency_checks.append(character_consistency)
                
                print(f"     Quality: {validation.get('quality_score', 0):.3f}, "
                      f"Consistency: {character_consistency['score']:.3f}")
            
            generation_time = time.time() - start_time
            
            # Create unique batch ID
            batch_id = str(uuid.uuid4())
            
            # Store batch data for later confirmation (NO MEMORY UPDATES YET)
            batch_data = {
                'batch_id': batch_id,
                'scene_prompt': scene_prompt,
                'enhanced_prompt': enhanced_prompt,
                'character_name': character_name,
                'story_id': story_id,
                'candidates': candidates,
                'validations': candidate_validations,
                'character_consistency_checks': candidate_consistency_checks,
                'generation_time': generation_time,
                'timestamp': time.time(),
                'consistency_strength': consistency_strength,
                'num_candidates': num_candidates,
                'confirmed': False  # CRITICAL: Not confirmed yet
            }
            
            # Store in pending batches (NOT in story memory)
            self.pending_batches[batch_id] = batch_data
            
            print(f"✅ Generated {num_candidates} candidates successfully")
            print(f"⚡ Total Generation Time: {generation_time:.2f}s")
            print(f"📦 Batch ID: {batch_id} (awaiting selection)")
            print(f"⚠️  MEMORY NOT UPDATED - awaiting confirmation of selected image")
            
            # Calculate summary scores for user decision-making
            quality_scores = [v.get('quality_score', 0) for v in candidate_validations]
            consistency_scores = [c['score'] for c in candidate_consistency_checks]
            
            return {
                'batch_id': batch_id,
                'candidates': candidates,
                'validations': candidate_validations,
                'character_consistency_checks': candidate_consistency_checks,
                'enhanced_prompt': enhanced_prompt,
                'story_id': story_id,
                'character_name': character_name,
                'generation_time': generation_time,
                'summary_scores': {
                    'best_quality': max(quality_scores),
                    'avg_quality': sum(quality_scores) / len(quality_scores),
                    'best_consistency': max(consistency_scores),
                    'avg_consistency': sum(consistency_scores) / len(consistency_scores)
                },
                'pending_confirmation': True,
                'message': f"Generated {num_candidates} candidates. Select one to update character memory."
            }
            
        except Exception as e:
            print(f"❌ Error generating candidate batch: {e}")
            return None
    
    def confirm_selected_image(self, batch_id, selected_index):
        """
        Confirm selection and UPDATE character memory with the chosen image
        
        Args:
            batch_id: The batch ID from generate_consistent_scene_batch
            selected_index: Index (0-based) of the selected candidate
            
        Returns:
            dict: Confirmation result with updated memory
        """
        if batch_id not in self.pending_batches:
            return {'error': f'Batch {batch_id} not found or already confirmed'}
        
        batch_data = self.pending_batches[batch_id]
        
        if selected_index < 0 or selected_index >= len(batch_data['candidates']):
            return {'error': f'Invalid selection index {selected_index}'}
        
        if batch_data['confirmed']:
            return {'error': 'This batch has already been confirmed'}
        
        try:
            # Get the selected image and its data
            selected_image = batch_data['candidates'][selected_index]
            selected_validation = batch_data['validations'][selected_index]
            selected_consistency = batch_data['character_consistency_checks'][selected_index]
            
            story_id = batch_data['story_id']
            character_name = batch_data['character_name']
            story_session = self.story_sessions[story_id]
            
            print(f"🎯 Confirming selection: candidate {selected_index + 1}")
            print(f"   Quality Score: {selected_validation.get('quality_score', 0):.3f}")
            print(f"   Consistency Score: {selected_consistency['score']:.3f}")
            
            # NOW UPDATE CHARACTER MEMORY with the selected image
            character_features = self._extract_and_store_character_features(
                selected_image, character_name, story_session
            )
            
            # Store character embedding using enhanced manager
            self.consistency_manager.extract_character_embedding(selected_image, character_name, store=True)
            
            # Record this scene in the story session
            scene_data = {
                'prompt': batch_data['scene_prompt'],
                'enhanced_prompt': batch_data['enhanced_prompt'],
                'character_name': character_name,
                'image': selected_image,
                'validation': selected_validation,
                'consistency_score': selected_validation.get('quality_score', 0.5),
                'character_consistency': selected_consistency,
                'character_features': character_features,
                'generation_time': batch_data['generation_time'],
                'timestamp': time.time(),
                'consistency_strength': batch_data['consistency_strength'],
                'selected_from_batch': True,
                'batch_id': batch_id,
                'selected_index': selected_index,
                'total_candidates': batch_data['num_candidates']
            }
            
            # Update story session
            story_session['scenes'].append(scene_data)
            story_session['consistency_scores'].append(selected_validation.get('quality_score', 0.5))
            story_session['character_consistency_scores'].append(selected_consistency['score'])
            
            # Mark batch as confirmed
            batch_data['confirmed'] = True
            batch_data['selected_index'] = selected_index
            batch_data['confirmed_at'] = time.time()
            
            print(f"✅ Selection confirmed and memory updated!")
            print(f"🎭 Character memory now includes features from selected image")
            print(f"📊 Story session updated with confirmed scene")
            
            # Clean up old pending batches (optional - keep recent ones for reference)
            self._cleanup_old_pending_batches()
            
            return {
                'success': True,
                'batch_id': batch_id,
                'selected_index': selected_index,
                'selected_image': selected_image,
                'scene_data': scene_data,
                'character_features': character_features,
                'validation': selected_validation,
                'character_consistency': selected_consistency,
                'story_id': story_id,
                'message': f'Selection confirmed. Character memory updated with candidate {selected_index + 1}.'
            }
            
        except Exception as e:
            print(f"❌ Error confirming selection: {e}")
            return {'error': f'Error confirming selection: {e}'}
    
    def _cleanup_old_pending_batches(self, max_age_hours=24):
        """Clean up old pending batches to prevent memory bloat"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_batches = []
        for batch_id, batch_data in self.pending_batches.items():
            age = current_time - batch_data['timestamp']
            if age > max_age_seconds:
                old_batches.append(batch_id)
        
        for batch_id in old_batches:
            del self.pending_batches[batch_id]
        
        if old_batches:
            print(f"🧹 Cleaned up {len(old_batches)} old pending batches")
    
    def list_pending_batches(self):
        """List all pending batch selections"""
        if not self.pending_batches:
            print("📦 No pending batch selections")
            return []
        
        print("📦 Pending Batch Selections:")
        for batch_id, batch_data in self.pending_batches.items():
            if not batch_data['confirmed']:
                scene_prompt = batch_data['scene_prompt'][:50] + "..." if len(batch_data['scene_prompt']) > 50 else batch_data['scene_prompt']
                num_candidates = batch_data['num_candidates']
                character_name = batch_data['character_name']
                age_minutes = (time.time() - batch_data['timestamp']) / 60
                
                print(f"   🎬 {batch_id[:8]}... - {scene_prompt}")
                print(f"      Character: {character_name}, Candidates: {num_candidates}, Age: {age_minutes:.1f}min")
        
        return [bid for bid, bd in self.pending_batches.items() if not bd['confirmed']]
    
    def cancel_pending_batch(self, batch_id):
        """Cancel a pending batch selection"""
        if batch_id in self.pending_batches:
            del self.pending_batches[batch_id]
            print(f"❌ Cancelled pending batch: {batch_id}")
            return True
        return False
    
    # Keep the original single-image generation method for backward compatibility
    def generate_consistent_scene(self, scene_prompt, character_name="main_character", 
                                 story_id=None, consistency_strength=0.8, **kwargs):
        """
        LEGACY: Generate a single scene and immediately update memory
        
        Note: This bypasses the batch selection process. Use generate_consistent_scene_batch() 
        for better consistency control.
        """
        print("⚠️  Using legacy single-image generation (immediate memory update)")
        
        # Generate a single candidate
        batch_result = self.generate_consistent_scene_batch(
            scene_prompt, character_name, story_id, num_candidates=1, 
            consistency_strength=consistency_strength, **kwargs
        )
        
        if not batch_result:
            return None
        
        # Automatically confirm the single candidate
        confirmation = self.confirm_selected_image(batch_result['batch_id'], 0)
        
        if confirmation.get('success'):
            return {
                'image': confirmation['selected_image'],
                'validation': confirmation['validation'],
                'consistency_score': confirmation['validation'].get('quality_score', 0.5),
                'character_consistency': confirmation['character_consistency'],
                'story_id': confirmation['story_id'],
                'scene_data': confirmation['scene_data'],
                'generation_time': batch_result['generation_time'],
                'character_features': confirmation['character_features']
            }
        else:
            return None
    
    def _extract_and_store_character_features(self, image, character_name, story_session):
        """Extract and store character features for consistency tracking"""
        try:
            # Initialize character data if needed
            if character_name not in story_session['characters']:
                story_session['characters'][character_name] = {
                    'features': [],
                    'embeddings': [],
                    'first_appearance': len(story_session['scenes']),
                    'appearance_count': 0
                }
            
            character_data = story_session['characters'][character_name]
            character_data['appearance_count'] += 1
            
            # Use CLIP analysis if available
            clip_analysis = None
            if self.consistency_manager.clip_available:
                clip_analysis = self.consistency_manager.analyze_character_consistency(
                    image, f"character named {character_name}"
                )
            
            # Extract basic character features based on scene count
            scene_count = len(story_session['scenes'])
            basic_features = []
            
            if scene_count == 0:  # First appearance
                basic_features = [
                    f"consistent {character_name} character design",
                    f"same {character_name} facial features",
                    f"{character_name} consistent appearance"
                ]
            else:  # Subsequent appearances
                basic_features = [
                    f"same {character_name} as previous scenes",
                    f"consistent {character_name} throughout story",
                    f"maintained {character_name} character design"
                ]
            
            # Add scene-specific consistency features
            if clip_analysis and clip_analysis.get('available'):
                clip_score = clip_analysis['consistency_score']
                if clip_score > 0.7:
                    basic_features.append(f"high character consistency ({clip_score:.2f})")
                elif clip_score > 0.5:
                    basic_features.append(f"moderate character consistency")
                else:
                    basic_features.append(f"focus on character consistency")
            
            # Store features (keep only the most recent ones to avoid prompt bloat)
            character_data['features'].extend(basic_features)
            
            # Keep only the last 5 features to prevent prompt from getting too long
            if len(character_data['features']) > 5:
                character_data['features'] = character_data['features'][-5:]
            
            print(f"🎭 Updated character features for {character_name}:")
            for feature in basic_features:
                print(f"   ✅ {feature}")
                
            return basic_features
            
        except Exception as e:
            print(f"⚠️ Could not extract character features: {e}")
            return []
    
    # Keep all other methods unchanged...
    def get_story_report(self, story_id=None):
        """Get detailed consistency report for a story session"""
        story_id = story_id or self.current_story_id
        if not story_id or story_id not in self.story_sessions:
            return {'error': 'No story session found'}
        
        story_session = self.story_sessions[story_id]
        scores = story_session['consistency_scores']
        char_scores = story_session['character_consistency_scores']
        
        if not scores:
            return {'error': 'No scenes generated yet'}
        
        # Calculate consistency metrics
        avg_score = sum(scores) / len(scores)
        avg_char_score = sum(char_scores) / len(char_scores) if char_scores else 0
        
        # Determine trend
        if len(scores) > 1:
            recent_avg = sum(scores[-3:]) / min(3, len(scores))
            early_avg = sum(scores[:3]) / min(3, len(scores))
            if recent_avg > early_avg + 0.05:
                consistency_trend = "improving"
            elif recent_avg < early_avg - 0.05:
                consistency_trend = "declining"
            else:
                consistency_trend = "stable"
        else:
            consistency_trend = "insufficient data"
        
        # Determine consistency grade
        combined_score = (avg_score + avg_char_score) / 2 if char_scores else avg_score
        
        if combined_score >= 0.8:
            grade = "A (Excellent)"
        elif combined_score >= 0.7:
            grade = "B (Good)"
        elif combined_score >= 0.6:
            grade = "C (Fair)"
        else:
            grade = "D (Needs Improvement)"
        
        total_time = sum(scene.get('generation_time', 0) for scene in story_session['scenes'])
        
        return {
            'story_name': story_session['name'],
            'story_id': story_id,
            'total_scenes': len(story_session['scenes']),
            'characters': list(story_session['characters'].keys()),
            'average_consistency_score': avg_score,
            'average_character_consistency': avg_char_score,
            'combined_consistency_score': combined_score,
            'consistency_grade': grade,
            'consistency_trend': consistency_trend,
            'score_range': f"{min(scores):.3f} - {max(scores):.3f}",
            'character_score_range': f"{min(char_scores):.3f} - {max(char_scores):.3f}" if char_scores else "N/A",
            'total_generation_time': total_time,
            'avg_time_per_scene': total_time / len(scores) if scores else 0,
            'all_scores': scores,
            'all_character_scores': char_scores,
            'clip_available': self.consistency_manager.clip_available
        }
    
    def export_current_story(self, story_id=None):
        """Export story data for animation or external use"""
        story_id = story_id or self.current_story_id
        if not story_id or story_id not in self.story_sessions:
            return {'error': 'No story session found'}
        
        story_session = self.story_sessions[story_id]
        
        # Prepare export data
        export_data = {
            'story_metadata': {
                'name': story_session['name'],
                'style': story_session['style'],
                'created_at': story_session['created_at'],
                'total_scenes': len(story_session['scenes']),
                'total_characters': len(story_session['characters'])
            },
            'characters': {},
            'scenes': [],
            'consistency_report': self.get_story_report(story_id)
        }
        
        # Export character data
        for char_name, char_data in story_session['characters'].items():
            export_data['characters'][char_name] = {
                'appearance_count': char_data.get('appearance_count', 0),
                'first_appearance': char_data.get('first_appearance', 0),
                'recent_features': char_data.get('features', [])[-3:],
                'total_embeddings': len(char_data.get('embeddings', []))
            }
        
        # Export scene data
        for i, scene in enumerate(story_session['scenes']):
            scene_export = {
                'scene_number': i + 1,
                'prompt': scene['prompt'],
                'enhanced_prompt': scene['enhanced_prompt'],
                'character_name': scene['character_name'],
                'consistency_score': scene['consistency_score'],
                'character_consistency_score': scene.get('character_consistency', {}).get('score', 0),
                'character_consistency_status': scene.get('character_consistency', {}).get('consistent', True),
                'generation_time': scene['generation_time'],
                'timestamp': scene['timestamp'],
                'consistency_strength': scene.get('consistency_strength', 0.8),
                'character_features_used': scene.get('character_features', []),
                'selected_from_batch': scene.get('selected_from_batch', False),
                'total_candidates': scene.get('total_candidates', 1),
                'validation_summary': {
                    'quality_score': scene['validation'].get('quality_score', 0),
                    'suitable_for_autism': scene['validation'].get('educational_assessment', {}).get('suitable_for_autism', False),
                    'tifa_score': scene['validation'].get('tifa_result', {}).get('score', 0)
                }
            }
            export_data['scenes'].append(scene_export)
        
        print(f"📦 Exported story data for: {story_session['name']}")
        print(f"   📊 {len(export_data['scenes'])} scenes")
        print(f"   🎭 {len(export_data['characters'])} characters")
        print(f"   🎯 Consistency Grade: {export_data['consistency_report']['consistency_grade']}")
        
        return export_data
    
    def reset_story_session(self, story_id=None):
        """Reset or clear a story session"""
        story_id = story_id or self.current_story_id
        if story_id and story_id in self.story_sessions:
            del self.story_sessions[story_id]
            if story_id == self.current_story_id:
                self.current_story_id = None
            print(f"🔄 Reset story session: {story_id}")
            return True
        return False
    
    def list_active_stories(self):
        """List all active story sessions"""
        if not self.story_sessions:
            print("📚 No active story sessions")
            return []
        
        print("📚 Active Story Sessions:")
        for story_id, session in self.story_sessions.items():
            scene_count = len(session['scenes'])
            char_count = len(session['characters'])
            current_marker = " (CURRENT)" if story_id == self.current_story_id else ""
            print(f"   📖 {session['name']} - {scene_count} scenes, {char_count} characters{current_marker}")
        
        return list(self.story_sessions.keys())

