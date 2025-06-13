"""
Caption Consistency Analyzer - Extracted from original script
Compares BLIP captions to detect character inconsistencies and artifacts
"""


class CaptionConsistencyAnalyzer:
    """Compare BLIP captions to detect character inconsistencies and artifacts"""
    
    def __init__(self):
        print("📝 Loading Caption Consistency Analyzer...")
        
        # Store character traits from selected images
        self.character_traits = {
            "hair_color": None,
            "hair_style": None, 
            "clothing": [],
            "age_descriptors": [],
            "physical_features": [],
            "established_scene_elements": []
        }
        
        self.caption_history = []
        self.available = True
        print("✅ Caption Consistency Analyzer ready")
    
    def extract_character_features(self, caption):
        """Extract character features from BLIP caption using NLP"""
        features = {
            "hair_color": [],
            "hair_style": [],
            "clothing": [],
            "age_descriptors": [],
            "physical_features": [],
            "potential_artifacts": [],
            "scene_elements": []
        }
        
        caption_lower = caption.lower()
        
        # Hair colors
        hair_colors = ["brown", "black", "blonde", "red", "gray", "white", "dark", "light"]
        for color in hair_colors:
            if f"{color} hair" in caption_lower or f"{color}-haired" in caption_lower:
                features["hair_color"].append(color)
        
        # Hair styles  
        hair_styles = ["short", "long", "curly", "straight", "messy", "neat"]
        for style in hair_styles:
            if f"{style} hair" in caption_lower:
                features["hair_style"].append(style)
        
        # Clothing
        clothing_items = ["shirt", "uniform", "jacket", "sweater", "pajamas", "clothes", "t-shirt", "dress"]
        for item in clothing_items:
            if item in caption_lower:
                features["clothing"].append(item)
        
        # Age descriptors
        age_words = ["young", "little", "small", "child", "kid", "boy", "teenager"]
        for age in age_words:
            if age in caption_lower:
                features["age_descriptors"].append(age)
        
        # Physical features
        physical = ["glasses", "smile", "eyes", "face", "tall", "short"]
        for feature in physical:
            if feature in caption_lower:
                features["physical_features"].append(feature)
        
        # Scene elements
        scene_words = ["kitchen", "classroom", "cafeteria", "bedroom", "table", "desk", "bed"]
        for scene in scene_words:
            if scene in caption_lower:
                features["scene_elements"].append(scene)
        
        # Potential artifacts (things that shouldn't be there)
        artifacts = ["blurry", "distorted", "multiple", "two heads", "extra", "deformed", 
                    "weird", "strange", "duplicate", "floating", "cut off", "many people",
                    "crowd", "several", "group"]
        for artifact in artifacts:
            if artifact in caption_lower:
                features["potential_artifacts"].append(artifact)
        
        return features
    
    def update_character_profile(self, caption, is_first_image=False):
        """Update character profile from selected image"""
        features = self.extract_character_features(caption)
        
        if is_first_image:
            # Establish character traits from first image
            if features["hair_color"]:
                self.character_traits["hair_color"] = features["hair_color"][0]
                print(f"🎭 Established hair color: {features['hair_color'][0]}")
            
            if features["hair_style"]:
                self.character_traits["hair_style"] = features["hair_style"][0]
                print(f"🎭 Established hair style: {features['hair_style'][0]}")
            
            self.character_traits["clothing"] = features["clothing"]
            self.character_traits["age_descriptors"] = features["age_descriptors"]
            self.character_traits["physical_features"] = features["physical_features"]
            
            print(f"🎭 Character profile established from: {caption}")
        
        # Store caption in history
        self.caption_history.append({
            "caption": caption,
            "features": features,
            "is_reference": is_first_image
        })
    
    def analyze_consistency_and_artifacts(self, current_caption):
        """Analyze current caption for consistency issues and artifacts"""
        current_features = self.extract_character_features(current_caption)
        
        issues = {
            "character_inconsistencies": [],
            "artifacts_detected": [],
            "scene_issues": [],
            "severity": "low"  # low, medium, high
        }
        
        # Check for artifacts first
        if current_features["potential_artifacts"]:
            issues["artifacts_detected"] = current_features["potential_artifacts"]
            issues["severity"] = "high"
            print(f"🚨 Artifacts detected: {', '.join(current_features['potential_artifacts'])}")
        
        # Check character consistency (only if we have established traits)
        if self.character_traits["hair_color"] and current_features["hair_color"]:
            established_hair = self.character_traits["hair_color"]
            current_hair = current_features["hair_color"][0]
            
            if established_hair != current_hair:
                issues["character_inconsistencies"].append(
                    f"Hair color changed from {established_hair} to {current_hair}"
                )
                if issues["severity"] == "low":
                    issues["severity"] = "medium"
                print(f"⚠️ Hair color inconsistency: {established_hair} → {current_hair}")
        
        # Check for multiple people (should be single character)
        multi_person_indicators = ["multiple", "two", "several", "group", "crowd", "many people"]
        for indicator in multi_person_indicators:
            if indicator in current_caption.lower():
                issues["scene_issues"].append(f"Multiple people detected: {indicator}")
                if issues["severity"] == "low":
                    issues["severity"] = "medium"
                print(f"⚠️ Multiple people detected: {indicator}")
        
        return issues
    
    def generate_consistency_improvements(self, issues, original_prompt):
        """Generate specific prompt improvements based on detected issues"""
        improvements = {
            "positive_additions": [],
            "negative_additions": [],
            "specific_fixes": []
        }
        
        # Fix character inconsistencies
        for inconsistency in issues["character_inconsistencies"]:
            if "hair color" in inconsistency.lower():
                if self.character_traits["hair_color"]:
                    improvements["positive_additions"].append(f"{self.character_traits['hair_color']} hair")
                    improvements["specific_fixes"].append(f"Enforce {self.character_traits['hair_color']} hair")
        
        # Fix artifacts
        for artifact in issues["artifacts_detected"]:
            improvements["negative_additions"].append(artifact)
            improvements["specific_fixes"].append(f"Remove {artifact}")
        
        # Fix multiple people issues
        for scene_issue in issues["scene_issues"]:
            if "multiple" in scene_issue.lower() or "people" in scene_issue.lower():
                improvements["positive_additions"].append("single person, one character only")
                improvements["negative_additions"].extend(["multiple people", "crowd", "group", "several people"])
                improvements["specific_fixes"].append("Ensure single character focus")
        
        # Add character consistency terms
        if self.character_traits["hair_color"]:
            improvements["positive_additions"].append(f"consistent {self.character_traits['hair_color']}-haired character")
        
        improvements["positive_additions"].append("same character, consistent design")
        
        return improvements