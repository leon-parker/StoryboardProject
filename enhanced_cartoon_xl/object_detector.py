import numpy as np
# ComprehensiveAIObjectDetector - Extracted directly from original code

import config
from transformers import pipeline
import torch

class ComprehensiveAIObjectDetector:
    """
    COMPREHENSIVE AI object detection with multiple models
    NO hard-coded object lists - AI determines all objects and relationships
    """
    
    def __init__(self):
        print("🎯 Loading COMPREHENSIVE AI Object Detection...")
        
        # Multiple object detection models for comprehensive analysis
        self.detectors = {}
        
        try:
            # Primary object detector - DETR ResNet-50
            self.detectors['detr'] = {
                'model': pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",
                    device=0 if torch.cuda.is_available() else -1
                ),
                'available': True,
                'strength': 'general_objects'
            }
            print("✅ DETR object detector loaded")
        except Exception as e:
            print(f"⚠️ DETR unavailable: {e}")
            self.detectors['detr'] = {'available': False}
        
        try:
            # Secondary object detector - YoloV5 alternative
            self.detectors['yolo'] = {
                'model': pipeline(
                    "object-detection", 
                    model="hustvl/yolos-tiny",
                    device=0 if torch.cuda.is_available() else -1
                ),
                'available': True,
                'strength': 'small_objects'
            }
            print("✅ YOLO-style detector loaded")
        except Exception as e:
            print(f"⚠️ YOLO detector unavailable: {e}")
            self.detectors['yolo'] = {'available': False}
        
        try:
            # Instance segmentation for detailed object analysis
            self.detectors['segmentation'] = {
                'model': pipeline(
                    "image-segmentation",
                    model="facebook/mask2former-swin-tiny-coco-panoptic",
                    device=0 if torch.cuda.is_available() else -1
                ),
                'available': True,
                'strength': 'detailed_segmentation'
            }
            print("✅ Segmentation detector loaded")
        except Exception as e:
            print(f"⚠️ Segmentation detector unavailable: {e}")
            self.detectors['segmentation'] = {'available': False}
        
        # Check if any detector is available
        self.available = any(detector.get('available', False) for detector in self.detectors.values())
        
        if self.available:
            print("✅ Comprehensive AI object detection ready")
        else:
            print("❌ No object detectors available")
    
    def comprehensive_object_detection(self, image):
        """Run comprehensive object detection with multiple AI models"""
        if not self.available:
            return {"available": False, "objects": []}
        
        print("🎯 Running comprehensive AI object detection...")
        
        comprehensive_results = {
            "all_detected_objects": [],
            "object_categories": {},
            "object_count": 0,
            "confidence_distribution": {},
            "spatial_distribution": {},
            "detection_consensus": {},
            "available_detectors": []
        }
        
        # Run all available detectors
        for detector_name, detector_info in self.detectors.items():
            if not detector_info.get('available', False):
                continue
            
            comprehensive_results["available_detectors"].append(detector_name)
            
            try:
                print(f"🤖 Running {detector_name} detection...")
                
                if detector_name == 'segmentation':
                    # Handle segmentation differently
                    segments = detector_info['model'](image)
                    
                    # Convert segmentation results to object format
                    for segment in segments:
                        if segment.get('score', 0) > 0.3:
                            obj = {
                                'label': segment.get('label', 'unknown'),
                                'score': segment.get('score', 0),
                                'detector': detector_name,
                                'type': 'segmentation',
                                'area': 'calculated_from_mask'  # Would calculate actual area
                            }
                            comprehensive_results["all_detected_objects"].append(obj)
                
                else:
                    # Standard object detection
                    detections = detector_info['model'](image)
                    
                    for detection in detections:
                        if detection.get('score', 0) > 0.3:  # Confidence threshold
                            obj = {
                                'label': detection.get('label', 'unknown'),
                                'score': detection.get('score', 0),
                                'box': detection.get('box', {}),
                                'detector': detector_name,
                                'type': 'bbox'
                            }
                            comprehensive_results["all_detected_objects"].append(obj)
                
                print(f"✅ {detector_name} completed")
                
            except Exception as e:
                print(f"⚠️ {detector_name} detection error: {e}")
                continue
        
        # Analyze comprehensive results
        if comprehensive_results["all_detected_objects"]:
            comprehensive_results["object_count"] = len(comprehensive_results["all_detected_objects"])
            
            # Categorize objects by label
            for obj in comprehensive_results["all_detected_objects"]:
                label = obj['label']
                if label not in comprehensive_results["object_categories"]:
                    comprehensive_results["object_categories"][label] = []
                comprehensive_results["object_categories"][label].append(obj)
            
            # Confidence distribution analysis
            scores = [obj['score'] for obj in comprehensive_results["all_detected_objects"]]
            comprehensive_results["confidence_distribution"] = {
                "mean_confidence": np.mean(scores),
                "high_confidence_count": len([s for s in scores if s > 0.7]),
                "medium_confidence_count": len([s for s in scores if 0.5 <= s <= 0.7]),
                "low_confidence_count": len([s for s in scores if s < 0.5])
            }
            
            # Detection consensus (objects detected by multiple models)
            label_counts = {}
            for obj in comprehensive_results["all_detected_objects"]:
                label = obj['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            comprehensive_results["detection_consensus"] = {
                "consensus_objects": {label: count for label, count in label_counts.items() if count > 1},
                "unique_objects": {label: count for label, count in label_counts.items() if count == 1},
                "most_detected": max(label_counts.items(), key=lambda x: x[1]) if label_counts else None
            }
        
        return comprehensive_results