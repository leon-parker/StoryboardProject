# AIBackgroundAnalyzer - Extracted directly from original code

import config
from object_detector import ComprehensiveAIObjectDetector
from transformers import pipeline
import torch
import numpy as np

class AIBackgroundAnalyzer:
    """
    ENHANCED AI background complexity analysis with comprehensive object detection
    NO hard-coded object lists - AI determines scene complexity dynamically
    """
    
    def __init__(self):
        print("🧮 Loading ENHANCED AI Background Analysis...")
        
        # Initialize comprehensive object detector
        self.object_detector = ComprehensiveAIObjectDetector()
        
        # Load depth estimation for spatial analysis
        try:
            self.depth_estimator = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if torch.cuda.is_available() else -1
            )
            self.depth_available = True
            print("✅ AI depth estimation loaded")
        except Exception as e:
            print(f"⚠️ Depth estimation unavailable: {e}")
            self.depth_available = False
        
        # Load scene classification for context
        try:
            self.scene_classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            self.scene_classification_available = True
            print("✅ Scene classifier loaded")
        except Exception as e:
            print(f"⚠️ Scene classifier unavailable: {e}")
            self.scene_classification_available = False
        
        self.available = self.object_detector.available
        
        if self.available:
            print("✅ Enhanced AI background analyzer loaded successfully")
        else:
            print("❌ Enhanced AI background analyzer unavailable")
    
    
    def enhanced_ai_analyze_background_complexity(self, image):
        """ENHANCED AI background analysis with comprehensive object detection"""
        if not self.available:
            return {"available": False}
        
        try:
            complexity_analysis = {
                "comprehensive_objects": {},
                "total_object_count": 0,
                "complexity_score": 0.5,
                "spatial_analysis": {},
                "scene_context": {},
                "autism_background_score": 0.5,
                "ai_recommendations": [],
                "detection_details": {}
            }
            
            # Comprehensive AI object detection
            print("🤖 Running comprehensive AI object detection...")
            object_results = self.object_detector.comprehensive_object_detection(image)
            complexity_analysis["comprehensive_objects"] = object_results
            complexity_analysis["total_object_count"] = object_results.get("object_count", 0)
            
            # AI scene classification for context
            if self.scene_classification_available:
                print("🤖 AI scene classification...")
                try:
                    scene_results = self.scene_classifier(image)
                    complexity_analysis["scene_context"] = {
                        "primary_scene": scene_results[0] if scene_results else None,
                        "scene_alternatives": scene_results[1:3] if len(scene_results) > 1 else [],
                        "scene_confidence": scene_results[0]['score'] if scene_results else 0
                    }
                except Exception as e:
                    print(f"Scene classification error: {e}")
                    complexity_analysis["scene_context"] = {"error": str(e)}
            
            # Enhanced AI complexity scoring based on comprehensive object detection
            object_count = complexity_analysis["total_object_count"]
            
            if object_count == 0:
                complexity_score = 0.1  # Very simple
                autism_score = 0.95    # Excellent for autism
                complexity_analysis["ai_recommendations"].append("AI: No objects detected - perfect minimal background for autism focus")
                
            elif object_count <= 2:
                complexity_score = 0.2  # Very simple
                autism_score = 0.9     # Excellent for autism
                complexity_analysis["ai_recommendations"].append("AI: Minimal objects - excellent simple background for autism learning")
                
            elif object_count <= 5:
                complexity_score = 0.4  # Simple
                autism_score = 0.8     # Good for autism
                complexity_analysis["ai_recommendations"].append("AI: Few objects - good simple background for autism learning")
                
            elif object_count <= 10:
                complexity_score = 0.6  # Moderate
                autism_score = 0.5     # Moderate for autism
                complexity_analysis["ai_recommendations"].append("AI: Moderate object count - may distract some autism learners")
                
            elif object_count <= 15:
                complexity_score = 0.8  # Complex
                autism_score = 0.3     # Poor for autism
                complexity_analysis["ai_recommendations"].append("AI: Many objects - likely to distract autism learners")
                
            else:
                complexity_score = 1.0  # Very complex
                autism_score = 0.1     # Very poor for autism
                complexity_analysis["ai_recommendations"].append("AI: Too many objects - will overwhelm autism learners")
            
            # Adjust scores based on object confidence and consensus
            if object_results.get("available", False):
                confidence_dist = object_results.get("confidence_distribution", {})
                high_conf_count = confidence_dist.get("high_confidence_count", 0)
                total_detections = object_results.get("object_count", 1)
                
                # If many low-confidence detections, reduce complexity penalty
                if total_detections > 0:
                    confidence_ratio = high_conf_count / total_detections
                    if confidence_ratio < 0.5:  # Many uncertain detections
                        complexity_score *= 0.8  # Reduce complexity penalty
                        autism_score = min(0.95, autism_score + 0.1)  # Slightly improve autism score
                        complexity_analysis["ai_recommendations"].append("AI: Many uncertain object detections - background may be simpler than initially assessed")
                
                # Consensus analysis
                consensus = object_results.get("detection_consensus", {})
                consensus_objects = consensus.get("consensus_objects", {})
                if len(consensus_objects) > len(object_results.get("object_categories", {})) * 0.8:
                    complexity_analysis["ai_recommendations"].append("AI: High detection consensus - reliable object assessment")
                else:
                    complexity_analysis["ai_recommendations"].append("AI: Mixed detection consensus - object assessment may vary")
            
            complexity_analysis["complexity_score"] = complexity_score
            complexity_analysis["autism_background_score"] = autism_score
            
            # Enhanced AI spatial analysis with depth
            if self.depth_available:
                try:
                    print("🤖 AI analyzing enhanced spatial depth...")
                    depth_result = self.depth_estimator(image)
                    depth_array = np.array(depth_result["depth"])
                    
                    # Enhanced depth analysis
                    depth_variance = np.var(depth_array)
                    depth_range = np.max(depth_array) - np.min(depth_array)
                    depth_gradients = np.gradient(depth_array)
                    gradient_magnitude = np.mean(np.abs(depth_gradients))
                    
                    spatial_complexity = "low"
                    if depth_variance > 0.15 or gradient_magnitude > 0.1:
                        spatial_complexity = "high"
                        complexity_analysis["ai_recommendations"].append("AI: Complex spatial depth - may confuse autism learners")
                    elif depth_variance > 0.05:
                        spatial_complexity = "moderate"
                        complexity_analysis["ai_recommendations"].append("AI: Moderate spatial depth - acceptable for autism learning")
                    else:
                        complexity_analysis["ai_recommendations"].append("AI: Simple spatial depth - excellent for autism focus")
                    
                    complexity_analysis["spatial_analysis"] = {
                        "depth_variance": float(depth_variance),
                        "depth_range": float(depth_range),
                        "gradient_magnitude": float(gradient_magnitude),
                        "spatial_complexity": spatial_complexity,
                        "ai_assessment": f"Spatial complexity: {spatial_complexity}"
                    }
                    
                    # Adjust autism score based on spatial complexity
                    if spatial_complexity == "high":
                        autism_score *= 0.8
                    elif spatial_complexity == "moderate":
                        autism_score *= 0.9
                    
                    complexity_analysis["autism_background_score"] = autism_score
                    
                except Exception as e:
                    print(f"Enhanced depth analysis error: {e}")
                    complexity_analysis["spatial_analysis"] = {"error": str(e)}
            
            # AI category-based insights
            if object_results.get("available", False):
                object_categories = object_results.get("object_categories", {})
                unique_categories = len(object_categories)
                
                if unique_categories <= 2:
                    complexity_analysis["ai_recommendations"].append("AI: Few object types - maintains clear focus")
                elif unique_categories <= 5:
                    complexity_analysis["ai_recommendations"].append("AI: Moderate object variety - manageable for autism learning")
                else:
                    complexity_analysis["ai_recommendations"].append("AI: Many object types - may scatter attention for autism learners")
                
                # Analyze specific object patterns that might affect autism learning
                detected_labels = list(object_categories.keys())
                
                # Check for potentially distracting elements (AI determines this dynamically)
                potentially_distracting = []
                for label in detected_labels:
                    # AI can identify distracting elements based on label semantics
                    if any(word in label.lower() for word in ['busy', 'crowd', 'multiple', 'many', 'complex']):
                        potentially_distracting.append(label)
                
                if potentially_distracting:
                    complexity_analysis["ai_recommendations"].append(f"AI: Potentially distracting elements detected: {', '.join(potentially_distracting)}")
            
            # Store detailed detection information
            complexity_analysis["detection_details"] = {
                "detectors_used": object_results.get("available_detectors", []),
                "detection_consensus": object_results.get("detection_consensus", {}),
                "confidence_analysis": object_results.get("confidence_distribution", {}),
                "total_detections": object_results.get("object_count", 0)
            }
            
            return complexity_analysis
            
        except Exception as e:
            print(f"Enhanced AI background analysis error: {e}")
            return {"available": False, "error": str(e)}
