"""
storyboard_creator.py - Smart storyboard creator that extracts meaningful labels

Works with any image generation prompts to create autism-friendly labels.
"""

import os
import re
from PIL import Image, ImageDraw, ImageFont


def extract_meaningful_label(prompt: str, max_words: int = 3) -> str:
    """Extract the most meaningful label from any prompt"""
    
    # Step 1: Clean the prompt
    # Remove weight syntax (word:1.5) and similar
    cleaned = re.sub(r'\([^)]*:[0-9.]+\)', '', prompt)
    # Remove CAPITALIZED emphasis
    cleaned = re.sub(r'\b[A-Z]{2,}\b', '', cleaned)
    # Remove extra parentheses and commas
    cleaned = re.sub(r'[(),]', ' ', cleaned)
    # Clean multiple spaces
    cleaned = ' '.join(cleaned.split())
    
    # Convert to lowercase for matching
    text = cleaned.lower().strip()
    
    # Step 2: Check for character references first
    if any(word in text for word in ['reference', 'character sheet', 'portrait', 't-pose']):
        # Extract character name from the original prompt
        name_match = re.search(r'\b(Alex|Emma|[A-Z][a-z]+)\b', prompt)
        if name_match:
            return name_match.group(1)
        return "Character"
    
    # Step 3: Priority emotion/state detection
    emotions_states = [
        # Sadness variations
        (['crying', 'sobbing', 'tears', 'weeping'], 'Feeling Sad'),
        (['sad', 'lonely', 'melancholic', 'devastated', 'unhappy'], 'Feeling Sad'),
        
        # Thinking variations
        (['thinking', 'pondering', 'contemplating', 'deep in thought'], 'Thinking'),
        (['thoughtful', 'pensive', 'reflecting'], 'Thinking'),
        
        # Happiness
        (['happy', 'joyful', 'cheerful', 'excited', 'smiling'], 'Feeling Happy'),
        
        # Other emotions
        (['curious', 'wondering', 'interested'], 'Being Curious'),
        (['worried', 'anxious', 'nervous', 'concerned'], 'Feeling Worried'),
        (['angry', 'mad', 'frustrated', 'upset'], 'Feeling Angry'),
        
        # States
        (['alone', 'by himself', 'by herself', 'solitary'], 'Alone'),
        (['together', 'with each other'], 'Together'),
    ]
    
    for keywords, label in emotions_states:
        if any(keyword in text for keyword in keywords):
            return label
    
    # Step 4: Action detection
    actions = [
        # Greetings/Social
        (['high five', 'high-five', 'highfive'], 'High Five!'),
        (['waving', 'wave hello', 'greeting'], 'Waving Hello'),
        (['shaking hands', 'handshake'], 'Handshake'),
        
        # Daily activities
        (['waking up', 'wake up', 'getting up'], 'Waking Up'),
        (['brushing teeth', 'brush teeth', 'toothbrush'], 'Brushing Teeth'),
        (['getting dressed', 'putting on clothes', 'dressing'], 'Getting Dressed'),
        (['eating breakfast', 'having breakfast', 'breakfast'], 'Eating Breakfast'),
        (['ready for school', 'prepared for school'], 'Ready for School'),
        
        # Play/Social activities
        (['playing', 'play together', 'playing with'], 'Playing Together'),
        (['sharing', 'share', 'giving'], 'Sharing'),
        (['helping', 'help', 'assisting'], 'Helping'),
        (['talking', 'speaking', 'conversation'], 'Talking'),
        (['reading', 'looking at book'], 'Reading'),
        (['drawing', 'coloring', 'painting'], 'Drawing'),
        
        # Movement
        (['walking towards', 'approaching', 'going to'], 'Walking Over'),
        (['running', 'jogging'], 'Running'),
        (['sitting', 'seated'], 'Sitting'),
        (['standing', 'stand'], 'Standing'),
    ]
    
    for keywords, label in actions:
        if any(keyword in text for keyword in keywords):
            return label
    
    # Step 5: Look for -ing verbs as actions
    ing_verbs = re.findall(r'\b(\w+ing)\b', text)
    if ing_verbs:
        # Filter out non-action words
        skip = ['wearing', 'showing', 'having', 'being', 'displaying', 'featuring']
        action_verbs = [v for v in ing_verbs if v not in skip]
        if action_verbs:
            # Take the first meaningful action
            verb = action_verbs[0]
            # Special formatting for common verbs
            if verb == 'looking':
                # Check what they're looking at
                if 'curious' in text:
                    return 'Looking Curious'
                elif 'sad' in text:
                    return 'Looking Sad'
                else:
                    return 'Looking'
            else:
                return verb.capitalize()
    
    # Step 6: Extract key descriptive phrases
    if 'looks up' in text:
        return 'Looking Up'
    if 'at desk' in text or 'at table' in text:
        return 'At Desk'
    if 'in bed' in text:
        return 'In Bed'
    
    # Step 7: Last resort - find the subject
    # Look for "boy/girl doing X" patterns
    subject_match = re.search(r'(boy|girl|child|person|character)\s+(\w+ing)', text)
    if subject_match:
        return subject_match.group(2).capitalize()
    
    # Final fallback
    return "Scene"


class StoryboardCreator:
    """Smart storyboard creator that works with any prompts"""
    
    def __init__(self, img_size=400, padding=40):
        self.img_size = img_size
        self.padding = padding
        self.images = []
        
        # Load fonts
        self._load_fonts()
    
    def _load_fonts(self):
        """Try to load nice fonts, fallback to default"""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        
        self.font = None
        self.title_font = None
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    self.font = ImageFont.truetype(font_path, 20)
                    self.title_font = ImageFont.truetype(font_path, 32)
                    break
                except:
                    continue
        
        if not self.font:
            self.font = ImageFont.load_default()
            self.title_font = self.font
    
    def add_image(self, image_path: str, prompt: str, label: str = None):
        """Add image with smart label extraction"""
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return
        
        img = Image.open(image_path)
        
        # Use provided label or extract from prompt
        if label is None:
            label = extract_meaningful_label(prompt)
        
        self.images.append((img, label))
        print(f"Added: {os.path.basename(image_path)} → '{label}'")
    
    def create(self, output_path: str, title: str = "Storyboard", cols: int = 3):
        """Create and save the storyboard"""
        
        if not self.images:
            print("No images to create storyboard!")
            return None
        
        # Calculate dimensions
        num_images = len(self.images)
        rows = (num_images + cols - 1) // cols
        
        canvas_width = cols * self.img_size + (cols + 1) * self.padding
        canvas_height = rows * (self.img_size + 60) + (rows + 1) * self.padding + 80
        
        # Create canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)
        
        # Draw title
        draw.text((canvas_width // 2, 30), title, fill='black', font=self.title_font, anchor='mt')
        
        # Draw images with labels
        for i, (img, label) in enumerate(self.images):
            row = i // cols
            col = i % cols
            
            x = self.padding + col * (self.img_size + self.padding)
            y = 80 + row * (self.img_size + 60 + self.padding)
            
            # Resize and paste image
            img_resized = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (x, y))
            
            # Draw label
            label_x = x + self.img_size // 2
            label_y = y + self.img_size + 10
            draw.text((label_x, label_y), label, fill='black', font=self.font, anchor='mt')
        
        # Save
        canvas.save(output_path, quality=95)
        print(f"\n✅ Storyboard saved: {output_path}")
        print(f"  📐 Size: {canvas_width}x{canvas_height}px")
        print(f"  🖼️ Images: {len(self.images)}")
        
        return output_path


# Quick test function
def test_label_extraction():
    """Test the label extraction with various prompts"""
    test_prompts = [
        "Alex, a cheerful 6-year-old boy with black hair, character reference sheet",
        "(CRYING EMMA:1.6), (SO SAD:1.5), sitting alone at her desk",
        "(THINKING ALEX:1.8), (DEEP IN THOUGHT:1.7), hand on chin thinking pose",
        "Alex walking towards Emma at her desk while Emma looks up",
        "both raising their right hands up high to do a high five together",
        "Alex and Emma playing with educational toys together",
        "Alex and Emma sharing books and educational materials",
        "boy waking up in bed yawning, tired eyes, stretching",
        "the same boy brushing his teeth with a toothbrush",
    ]
    
    print("Testing label extraction:")
    print("-" * 50)
    for prompt in test_prompts:
        label = extract_meaningful_label(prompt)
        print(f"Prompt: {prompt[:60]}...")
        print(f"Label: {label}\n")


if __name__ == "__main__":
    test_label_extraction()