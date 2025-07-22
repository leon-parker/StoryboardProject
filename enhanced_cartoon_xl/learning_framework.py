# Learning Framework - Extracted from your original learning system
# Contains the core learning algorithms and improvement strategies
import numpy as np
import config

class AILearningFramework:
    """
    Core AI learning framework that analyzes failures and improves prompts
    Extracted from your original generate_truly_learning_storyboard method
    """
    
    def __init__(self, validator, ai_refiner):
        self.validator = validator
        self.ai_refiner = ai_refiner
        self.learning_history = []
        
    def analyze_learning_needs(self, validation_result):
        """
        Analyze what the AI needs to learn from validation results
        """
        learning_plan = {
            "needs_learning": False,
            "learning_areas": [],
            "prompt_improvements": [],
            "negative_improvements": [],
            "parameter_adjustments": {},
            "priority": "medium"
        }
        
        # Analyze TIFA faithfulness learning needs
        tifa_score = validation_result.get('tifa_result', {}).get('score', 0.5)
        if tifa_score < 0.6:
            learning_plan["needs_learning"] = True
            learning_plan["learning_areas"].append("prompt_faithfulness")
            learning_plan["prompt_improvements"].extend([
                "highly accurate representation",
                "exactly as described",
                "faithful to prompt"
            ])
            learning_plan["priority"] = "high"
        
        # Analyze emotional learning needs
        emotion_score = validation_result.get('emotion_analysis', {}).get('autism_appropriateness_score', 0.5)
        if emotion_score < 0.5:
            learning_plan["needs_learning"] = True
            learning_plan["learning_areas"].append("emotional_appropriateness")
            learning_plan["prompt_improvements"].extend([
                "calm expression",
                "gentle demeanor", 
                "friendly face",
                "peaceful emotion"
            ])
            learning_plan["negative_improvements"].extend([
                "scary expression",
                "angry face",
                "intense emotions"
            ])
        
        # Analyze background complexity learning needs
        bg_score = validation_result.get('background_analysis', {}).get('autism_background_score', 0.5)
        if bg_score < 0.5:
            learning_plan["needs_learning"] = True
            learning_plan["learning_areas"].append("background_simplification")
            learning_plan["prompt_improvements"].extend([
                "simple background",
                "minimal objects",
                "clean environment",
                "uncluttered scene"
            ])
            learning_plan["negative_improvements"].extend([
                "cluttered background",
                "busy scene",
                "many objects",
                "distracting elements"
            ])
        
        # Analyze character clarity learning needs
        face_count = validation_result.get('face_count', 0)
        if face_count == 0:
            learning_plan["needs_learning"] = True
            learning_plan["learning_areas"].append("character_visibility")
            learning_plan["prompt_improvements"].extend([
                "clear visible character",
                "distinct facial features",
                "well-defined person"
            ])
            learning_plan["negative_improvements"].extend([
                "hidden face",
                "no visible character",
                "obscured person"
            ])
        elif face_count > 2:
            learning_plan["needs_learning"] = True
            learning_plan["learning_areas"].append("character_focus")
            learning_plan["prompt_improvements"].extend([
                "single character focus",
                "one person only",
                "individual subject"
            ])
            learning_plan["negative_improvements"].extend([
                "multiple people",
                "crowd",
                "many faces",
                "group scene"
            ])
        
        # Parameter learning needs
        overall_score = validation_result.get('quality_score', 0.5)
        if overall_score < 0.4:
            learning_plan["parameter_adjustments"]["guidance_scale_increase"] = 0.5
            learning_plan["parameter_adjustments"]["steps_increase"] = 5
        
        return learning_plan
    
    def apply_learning_improvements(self, current_prompt, current_negative, learning_plan, generation_settings):
        """
        Apply learning improvements to prompts and parameters
        """
        improved_prompt = current_prompt
        improved_negative = current_negative
        improvements_made = []
        
        # Apply prompt improvements
        if learning_plan["prompt_improvements"]:
            unique_improvements = list(dict.fromkeys(learning_plan["prompt_improvements"]))[:3]
            additions = ", ".join(unique_improvements)
            improved_prompt = f"{improved_prompt}, {additions}"
            improvements_made.append(f"Enhanced prompt with: {additions}")
        
        # Apply negative prompt improvements
        if learning_plan["negative_improvements"]:
            unique_negatives = list(dict.fromkeys(learning_plan["negative_improvements"]))[:3]
            additions = ", ".join(unique_negatives)
            improved_negative = f"{improved_negative}, {additions}"
            improvements_made.append(f"Enhanced negative with: {additions}")
        
        # Apply parameter adjustments
        for param, adjustment in learning_plan["parameter_adjustments"].items():
            if param == "guidance_scale_increase":
                new_value = min(8.0, generation_settings['guidance_scale'] + adjustment)
                generation_settings['guidance_scale'] = new_value
                improvements_made.append(f"Increased guidance scale to {new_value}")
            elif param == "steps_increase":
                new_value = min(40, generation_settings['num_inference_steps'] + adjustment)
                generation_settings['num_inference_steps'] = new_value
                improvements_made.append(f"Increased steps to {new_value}")
        
        return improved_prompt, improved_negative, improvements_made
    
    def track_learning_progress(self, learning_entry):
        """
        Track learning progress across attempts
        """
        self.learning_history.append(learning_entry)
        
        # Calculate learning metrics
        if len(self.learning_history) > 1:
            scores = [entry['score'] for entry in self.learning_history]
            learning_entry['score_improvement'] = scores[-1] - scores[0]
            learning_entry['learning_trend'] = 'improving' if scores[-1] > scores[-2] else 'declining'
        
        return learning_entry
    
    def generate_learning_report(self, best_score, target_score, original_prompt, final_prompt):
        """
        Generate comprehensive learning report
        """
        report = {
            "learning_success": best_score >= target_score,
            "final_score": best_score,
            "target_score": target_score,
            "total_attempts": len(self.learning_history),
            "score_progression": [entry['score'] for entry in self.learning_history],
            "learning_areas_addressed": [],
            "total_improvements": 0,
            "prompt_evolution": {
                "original": original_prompt,
                "final": final_prompt,
                "evolution_length": len(final_prompt) - len(original_prompt)
            },
            "learning_efficiency": best_score / len(self.learning_history) if self.learning_history else 0
        }
        
        # Analyze learning areas
        all_improvements = []
        for entry in self.learning_history:
            all_improvements.extend(entry.get('improvements_made', []))
        
        report["total_improvements"] = len([imp for imp in all_improvements if imp != "No specific refinements needed"])
        
        # Identify learning patterns
        learning_areas = set()
        for entry in self.learning_history:
            improvements = entry.get('improvements_made', [])
            for improvement in improvements:
                if "emotion" in improvement.lower():
                    learning_areas.add("emotional_appropriateness")
                elif "background" in improvement.lower():
                    learning_areas.add("background_simplification")
                elif "character" in improvement.lower():
                    learning_areas.add("character_clarity")
                elif "prompt" in improvement.lower():
                    learning_areas.add("prompt_faithfulness")
        
        report["learning_areas_addressed"] = list(learning_areas)
        
        return report

class LearningOptimizer:
    """
    Advanced learning optimizer that fine-tunes the learning process
    """
    
    def __init__(self):
        self.optimization_history = []
        self.best_strategies = {}
    
    def optimize_learning_strategy(self, validation_result, attempt_number):
        """
        Optimize learning strategy based on current performance
        """
        strategy = {
            "learning_aggressiveness": "moderate",
            "focus_areas": [],
            "prompt_weight": 0.7,
            "negative_weight": 0.3,
            "parameter_weight": 0.5
        }
        
        # Adjust strategy based on attempt number
        if attempt_number == 1:
            strategy["learning_aggressiveness"] = "conservative"
            strategy["prompt_weight"] = 0.8
        elif attempt_number == 2:
            strategy["learning_aggressiveness"] = "moderate"
        else:
            strategy["learning_aggressiveness"] = "aggressive"
            strategy["parameter_weight"] = 0.8
        
        # Focus on biggest issues first
        scores = {
            "tifa": validation_result.get('tifa_result', {}).get('score', 0.5),
            "emotion": validation_result.get('emotion_analysis', {}).get('autism_appropriateness_score', 0.5),
            "background": validation_result.get('background_analysis', {}).get('autism_background_score', 0.5),
            "overall": validation_result.get('quality_score', 0.5)
        }
        
        # Sort by lowest scores (biggest problems)
        sorted_issues = sorted(scores.items(), key=lambda x: x[1])
        strategy["focus_areas"] = [issue[0] for issue in sorted_issues[:2]]
        
        return strategy
    
    def calculate_learning_efficiency(self, learning_history):
        """
        Calculate how efficiently the AI is learning
        """
        if len(learning_history) < 2:
            return 0.5
        
        scores = [entry['score'] for entry in learning_history]
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        # Learning efficiency = average improvement per attempt
        efficiency = np.mean([imp for imp in improvements if imp > 0]) if any(imp > 0 for imp in improvements) else 0
        
        return min(1.0, max(0.0, efficiency * 2))  # Normalize to 0-1 range

def demo_learning_framework():
    """Demo the learning framework capabilities"""
    
    print("\n🧠 AI LEARNING FRAMEWORK DEMO")
    print("=" * 50)
    
    print("\n🎯 LEARNING CAPABILITIES:")
    print("✅ Automatic failure analysis")
    print("✅ Intelligent prompt improvements")
    print("✅ Dynamic parameter adjustments")
    print("✅ Learning progress tracking")
    print("✅ Strategy optimization")
    print("✅ Comprehensive reporting")
    
    print("\n💡 USAGE EXAMPLE:")
    print("learning_framework = AILearningFramework(validator, ai_refiner)")
    print("learning_plan = learning_framework.analyze_learning_needs(validation)")
    print("improved_prompt, improved_negative, improvements = learning_framework.apply_learning_improvements(...)")
    print("report = learning_framework.generate_learning_report(...)")
    
    print("\n🚀 LEARNING AREAS:")
    print("• Prompt Faithfulness - TIFA-based improvements")
    print("• Emotional Appropriateness - Autism-friendly emotions")
    print("• Background Simplification - Reduced complexity")
    print("• Character Clarity - Single character focus")
    print("• Parameter Optimization - Dynamic adjustments")
    
    print("\n📈 LEARNING METRICS:")
    print("• Score progression tracking")
    print("• Learning efficiency calculation") 
    print("• Strategy optimization")
    print("• Improvement categorization")
    print("• Success rate analysis")

if __name__ == "__main__":
    demo_learning_framework()