"""
Prompt Improver - Extracted from original script
Improve prompts based on CLIP/BLIP analysis with caption consistency
"""

import numpy as np
from caption_analyzer import CaptionConsistencyAnalyzer


class PromptImprover:
    """Improve prompts based on CLIP/BLIP analysis with caption consistency"""
    
    def __init__(self, clip_memory, captioner):
        self.clip_memory = clip_memory
        self.captioner = captioner
        self.caption_analyzer = CaptionConsistencyAnalyzer()
        
    def analyze_prompt_image_alignment(self, image, prompt, is_first_image=False):
        """Analyze how well image matches prompt with consistency checking"""
        # Get caption
        caption = self.captioner.generate_caption(image)
        
        # Calculate CLIP similarity
        clip_similarity = 0.5
        if self.clip_memory.available:
            prompt_embedding = self.clip_memory.get_text_embedding(prompt)
            image_embedding = self.clip_memory.get_image_embedding(image)
            
            if prompt_embedding is not None and image_embedding is not None:
                clip_similarity = np.dot(prompt_embedding.flatten(), image_embedding.flatten()) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(image_embedding)
                )
        
        # Update character profile and analyze consistency
        self.caption_analyzer.update_character_profile(caption, is_first_image)
        consistency_issues = self.caption_analyzer.analyze_consistency_and_artifacts(caption)
        
        return {
            "caption": caption,
            "clip_similarity": float(clip_similarity),
            "alignment_quality": "good" if clip_similarity > 0.7 else "moderate" if clip_similarity > 0.5 else "poor",
            "consistency_issues": consistency_issues,
            "character_established": is_first_image
        }
    
    def improve_prompts(self, original_prompt, analysis_result, current_negative=""):
        """Improve positive and negative prompts based on analysis and consistency"""
        improved_positive = original_prompt
        improved_negative = current_negative or "blurry, low quality, distorted, deformed"
        
        improvements = []
        
        # Handle consistency issues first (most important)
        consistency_issues = analysis_result.get("consistency_issues", {})
        if consistency_issues["severity"] != "low":
            consistency_fixes = self.caption_analyzer.generate_consistency_improvements(
                consistency_issues, original_prompt
            )
            
            # Add consistency-specific improvements
            for addition in consistency_fixes["positive_additions"]:
                if addition not in improved_positive.lower():
                    improved_positive = f"{improved_positive}, {addition}"
                    improvements.append(f"Consistency fix: {addition}")
            
            for negative_addition in consistency_fixes["negative_additions"]:
                if negative_addition not in improved_negative.lower():
                    improved_negative = f"{improved_negative}, {negative_addition}"
                    improvements.append(f"Negative consistency: {negative_addition}")
            
            # Log specific fixes
            for fix in consistency_fixes["specific_fixes"]:
                improvements.append(f"Applied: {fix}")
        
        # Standard CLIP-based improvements
        if analysis_result["clip_similarity"] < 0.6:
            clarity_terms = ["detailed", "clear", "well-defined", "accurate representation"]
            for term in clarity_terms:
                if term not in improved_positive.lower():
                    improved_positive = f"{improved_positive}, {term}"
                    improvements.append(f"Added clarity: {term}")
                    break
        
        # Add quality terms if not present
        quality_terms = ["high quality", "masterpiece", "best quality"]
        has_quality = any(term in improved_positive.lower() for term in quality_terms)
        if not has_quality:
            improved_positive = f"{improved_positive}, high quality, detailed"
            improvements.append("Added quality terms")
        
        # Improve negative prompt with standard terms
        common_negatives = ["blurry", "low quality", "distorted", "deformed", "ugly", "bad anatomy"]
        for neg_term in common_negatives:
            if neg_term not in improved_negative.lower():
                improved_negative = f"{improved_negative}, {neg_term}"
                improvements.append(f"Enhanced negative: {neg_term}")
        
        return {
            "positive": improved_positive,
            "negative": improved_negative,
            "improvements": improvements,
            "consistency_issues_found": consistency_issues["severity"] != "low"
        }