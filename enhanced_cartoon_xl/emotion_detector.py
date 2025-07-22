import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from transformers import pipeline
import torch

class AIEmotionDetector:
    """
    PURE AI emotion detection - Uses text-based emotion consistency
    Compares emotions between prompt and generated image caption
    """
    
    def __init__(self):
        print("👥 Loading PURE AI Emotion Detection...")
        
        try:
            # Initialize MediaPipe for face detection (for face counting)
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=0.5
            )
            
            # Load TEXT emotion classification pipeline (not image)
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",  # Changed from image-classification
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.emotion_ai_available = True
                print("✅ AI emotion classifier loaded")
            except Exception as e:
                print(f"⚠️ Advanced emotion AI unavailable: {e}")
                self.emotion_ai_available = False
            
            self.available = True
            print("✅ AI emotion detection loaded successfully")
            
        except Exception as e:
            print(f"⚠️ AI emotion detection unavailable: {e}")
            self.available = False
    
    def analyze_emotion_consistency(self, image, prompt, image_caption=None):
        """
        Check if emotions in image are consistent with prompt expectations
        Uses text-based emotion analysis for reliability
        """
        if not self.available:
            return {"available": False, "consistency_score": 0.5}
        
        try:
            # Count faces for context
            face_count = self.count_faces(image)
            
            result = {
                "face_count": face_count,
                "prompt_emotions": {},
                "caption_emotions": {},
                "consistency_score": 0.5,
                "autism_appropriateness": 0.6,
                "recommendations": []
            }
            
            if self.emotion_ai_available:
                # Analyze prompt emotions
                try:
                    prompt_analysis = self.emotion_classifier(prompt)
                    result["prompt_emotions"] = self.process_emotion_results(prompt_analysis)
                except Exception as e:
                    print(f"Prompt emotion analysis error: {e}")
                
                # Analyze image caption emotions (if provided)
                if image_caption:
                    try:
                        caption_analysis = self.emotion_classifier(image_caption)
                        result["caption_emotions"] = self.process_emotion_results(caption_analysis)
                        
                        # Calculate consistency
                        result["consistency_score"] = self.calculate_consistency(
                            result["prompt_emotions"], 
                            result["caption_emotions"]
                        )
                    except Exception as e:
                        print(f"Caption emotion analysis error: {e}")
                
                # Determine autism appropriateness
                result["autism_appropriateness"] = self.assess_autism_appropriateness(
                    result["prompt_emotions"], 
                    result["caption_emotions"]
                )
                
                # Generate recommendations
                result["recommendations"] = self.generate_recommendations(result)
            
            return result
            
        except Exception as e:
            print(f"Emotion consistency analysis error: {e}")
            return {"available": False, "error": str(e), "consistency_score": 0.5}
    
    def count_faces(self, image):
        """Count faces in image for context"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            face_results = self.face_detection.process(rgb_image)
            return len(face_results.detections) if face_results.detections else 0
            
        except Exception as e:
            print(f"Face counting error: {e}")
            return 0
    
    def process_emotion_results(self, emotion_results):
        """Process emotion classification results"""
        emotions = {}
        if emotion_results:
            for result in emotion_results:
                label = result['label'].lower()
                score = result['score']
                emotions[label] = score
        return emotions
    
    def calculate_consistency(self, prompt_emotions, caption_emotions):
        """Calculate emotion consistency between prompt and caption"""
        if not prompt_emotions or not caption_emotions:
            return 0.6  # Neutral when can't compare
        
        # Find overlapping emotions and calculate similarity
        common_emotions = set(prompt_emotions.keys()) & set(caption_emotions.keys())
        
        if not common_emotions:
            return 0.5  # No common emotions detected
        
        # Calculate weighted consistency based on emotion scores
        consistency_scores = []
        for emotion in common_emotions:
            prompt_score = prompt_emotions[emotion]
            caption_score = caption_emotions[emotion]
            # Similarity based on how close the scores are
            similarity = 1 - abs(prompt_score - caption_score)
            consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def assess_autism_appropriateness(self, prompt_emotions, caption_emotions):
        """Assess if emotions are appropriate for autism-friendly content"""
        # Autism-friendly emotions (generally calm, positive, not overwhelming)
        autism_friendly = ['joy', 'calm', 'neutral', 'contentment', 'peace']
        potentially_challenging = ['anger', 'fear', 'sadness', 'surprise', 'disgust']
        
        score = 0.6  # Default neutral
        
        # Check prompt emotions
        for emotion, confidence in prompt_emotions.items():
            if emotion in autism_friendly:
                score += confidence * 0.2
            elif emotion in potentially_challenging:
                score -= confidence * 0.1
        
        # Check caption emotions  
        for emotion, confidence in caption_emotions.items():
            if emotion in autism_friendly:
                score += confidence * 0.2
            elif emotion in potentially_challenging:
                score -= confidence * 0.1
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def generate_recommendations(self, analysis):
        """Generate autism-friendly recommendations"""
        recommendations = []
        
        consistency = analysis["consistency_score"]
        appropriateness = analysis["autism_appropriateness"]
        
        if consistency > 0.7:
            recommendations.append("✅ Emotions are consistent between prompt and image")
        elif consistency > 0.5:
            recommendations.append("⚠️ Some emotional inconsistency detected")
        else:
            recommendations.append("❌ Significant emotional mismatch between prompt and image")
        
        if appropriateness > 0.7:
            recommendations.append("✅ Emotions are appropriate for autism learning")
        elif appropriateness > 0.5:
            recommendations.append("⚠️ Emotions may need adjustment for autism learners")
        else:
            recommendations.append("❌ Emotions may be challenging for autism learners")
        
        return recommendations

    # Legacy method for backward compatibility
    def ai_analyze_facial_emotions(self, image):
        """Legacy method - returns simple emotion analysis"""
        face_count = self.count_faces(image)
        return {
            "face_count": face_count,
            "autism_appropriateness_score": 0.6,
            "available": self.available
        }