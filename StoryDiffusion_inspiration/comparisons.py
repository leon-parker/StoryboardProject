"""
SIMPLE 4-SCENARIO MODEL COMPARISON
Just generates the 4 specific scenarios you want and creates the comparison grid
"""

import os
import torch
import time
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc

class SimpleModelComparison:
    def __init__(self, device="cuda"):
        self.device = device
        self.current_pipeline = None
        self.current_model_name = None
        self.compel = None
        self.comparison_seed = 42
        
        # Your 4 specific scenarios
        self.alex_character = "6-year-old boy Alex, black hair, brown eyes"
        self.pajama_details = "blue star-patterned pajamas, light blue cotton fabric with white five-pointed stars, long-sleeved button-up pajama top with white buttons, matching pajama pants"
        self.uniform_details = "school uniform, crisp white button-up shirt with collar, dark navy blue tie with diagonal stripes, dark navy blue shorts with belt loops"
        self.style = "cartoon illustration, animated style, cel shading, cartoon character design, flat colors, NOT realistic, NOT photographic"
        
        self.scenarios = {
            "emotion_test_sleepy": {
                "prompt": f"{self.alex_character}, tired expression, rubbing eyes with hands, yawning, wearing {self.pajama_details}, {self.style}",
                "negative": "wide awake, energetic, multiple people, wrong clothing, adult, distorted, realistic, photorealistic, photograph, real person, live action, 3D render, hyperrealistic, cinematic, professional photography"
            },
            "environment_outdoor": {
                "prompt": f"{self.alex_character}, standing in school playground, swings and slides visible, sunny day, wearing {self.uniform_details}, {self.style}",
                "negative": "indoor setting, wrong environment, multiple people, wrong clothing, adult, distorted, realistic, photorealistic, photograph, real person, live action, 3D render, hyperrealistic, cinematic, professional photography"
            },
            "emotion_test_happy": {
                "prompt": f"{self.alex_character}, laughing happily, big smile, eyes closed with joy, wearing {self.pajama_details}, {self.style}",
                "negative": "sad, crying, angry, multiple people, wrong clothing, adult, distorted, realistic, photorealistic, photograph, real person, live action, 3D render, hyperrealistic, cinematic, professional photography"
            },
            "environment_indoor": {
                "prompt": f"{self.alex_character}, standing in living room, sofa and TV visible, cozy indoor lighting, wearing {self.uniform_details}, {self.style}",
                "negative": "outdoor setting, bedroom, kitchen, multiple people, wrong clothing, adult, distorted, realistic, photorealistic, photograph, real person, live action, 3D render, hyperrealistic, cinematic, professional photography"
            }
        }
        
        # Models to test
        self.models = {
            "realcartoon_xl_v7": "../models/realcartoonxl_v7.safetensors",
            "animagine_xl_v3": "cagliostrolab/animagine-xl-3.1",
            "counterfeit_xl": "gsdf/CounterfeitXL", 
            "dreamshaper_xl": "Lykon/dreamshaper-xl-1-0"
        }

    def setup_model(self, model_name):
        if model_name == self.current_model_name and self.current_pipeline:
            return True
            
        print(f"\n🔧 Loading {model_name}...")
        
        try:
            # Clear previous model
            if self.current_pipeline:
                del self.current_pipeline
                if self.compel:
                    del self.compel
                torch.cuda.empty_cache()
                gc.collect()
            
            model_path = self.models[model_name]
            
            # Load model
            if "/" in model_path and not os.path.exists(model_path):
                print(f"   📥 Downloading: {model_path}")
                self.current_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path, torch_dtype=torch.float16, use_safetensors=True
                ).to(self.device)
            else:
                print(f"   📁 Loading local: {model_path}")
                self.current_pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_path, torch_dtype=torch.float16, use_safetensors=True
                ).to(self.device)
            
            # Setup
            self.current_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.current_pipeline.scheduler.config
            )
            self.current_pipeline.enable_vae_tiling()
            self.current_pipeline.enable_model_cpu_offload()
            
            # Setup Compel
            self.compel = Compel(
                tokenizer=[self.current_pipeline.tokenizer, self.current_pipeline.tokenizer_2],
                text_encoder=[self.current_pipeline.text_encoder, self.current_pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
            
            self.current_model_name = model_name
            print(f"✅ {model_name} loaded")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return False

    def generate_image(self, prompt, negative_prompt):
        try:
            # Encode prompts
            positive_conditioning, positive_pooled = self.compel(prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)
            
            [positive_conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
            # Generate
            result = self.current_pipeline(
                prompt_embeds=positive_conditioning,
                pooled_prompt_embeds=positive_pooled,
                negative_prompt_embeds=negative_conditioning,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=1024,
                width=1024,
                generator=torch.Generator(self.device).manual_seed(self.comparison_seed)
            )
            
            return result.images[0]
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None

    def create_comparison_grid(self, output_dir):
        print("\n📋 Creating 4-scenario comparison grid...")
        
        # Grid settings with more left space
        padding = 40
        img_size = 300
        label_height = 60
        left_margin = 250  # Increased from 150 to 250 for more space
        
        # 4 models x 4 scenarios
        cols = 4  # models
        rows = 4  # scenarios
        
        canvas_width = left_margin + cols * img_size + (cols + 1) * padding
        canvas_height = rows * (img_size + label_height) + (rows + 1) * padding + 120
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        draw = ImageDraw.Draw(canvas)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 28)
            header_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
            label_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 14)  # Slightly smaller for row labels
        except:
            title_font = header_font = label_font = ImageFont.load_default()
        
        # Title
        title = "4-Scenario Model Comparison"
        draw.text((canvas_width // 2, 30), title, fill='black', font=title_font, anchor='mt')
        
        # Column headers (model names)
        model_names = ["RealCartoon XL v7", "Animagine XL v3", "Counterfeit XL", "DreamShaper XL"]
        for col, model_name in enumerate(model_names):
            x = left_margin + padding + col * (img_size + padding) + img_size // 2
            y = 70
            draw.text((x, y), model_name, fill='black', font=header_font, anchor='mt')
        
        # Row labels and images
        scenario_labels = [
            "Sleepy Emotion Test",
            "Outdoor Environment",
            "Happy Emotion Test", 
            "Indoor Environment"
        ]
        scenario_keys = list(self.scenarios.keys())
        model_keys = list(self.models.keys())
        
        for row, (scenario_key, scenario_label) in enumerate(zip(scenario_keys, scenario_labels)):
            # Row label with more space and better positioning
            label_x = left_margin - 20  # More margin from the images
            label_y = 120 + row * (img_size + label_height + padding) + img_size // 2
            
            # Multi-line text handling for longer labels
            if len(scenario_label) > 15:
                # Split longer labels into multiple lines
                words = scenario_label.split()
                if len(words) >= 2:
                    line1 = " ".join(words[:len(words)//2])
                    line2 = " ".join(words[len(words)//2:])
                    draw.text((label_x, label_y - 10), line1, fill='black', font=label_font, anchor='rm')
                    draw.text((label_x, label_y + 10), line2, fill='black', font=label_font, anchor='rm')
                else:
                    draw.text((label_x, label_y), scenario_label, fill='black', font=label_font, anchor='rm')
            else:
                draw.text((label_x, label_y), scenario_label, fill='black', font=label_font, anchor='rm')
            
            for col, model_key in enumerate(model_keys):
                image_path = os.path.join(output_dir, f"{scenario_key}_{model_key}.png")
                
                x = left_margin + padding + col * (img_size + padding)
                y = 120 + row * (img_size + label_height + padding)
                
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    canvas.paste(img_resized, (x, y))
                else:
                    # Missing image placeholder
                    draw.rectangle([x, y, x+img_size, y+img_size], outline='red', fill='lightgray')
                    draw.text((x + img_size//2, y + img_size//2), "Missing", fill='red', font=header_font, anchor='mm')
        
        # Save grid
        grid_path = os.path.join(output_dir, "4_scenario_comparison_grid.png")
        canvas.save(grid_path, quality=95)
        
        print(f"✅ Comparison grid saved: {grid_path}")
        return grid_path

def main():
    print("🎨 4-Scenario Model Comparison")
    print("=" * 40)
    
    output_dir = "4_scenario_results"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison = SimpleModelComparison()
    
    print("Generating images for 4 scenarios across 4 models...")
    print("Estimated time: ~16 minutes\n")
    
    successful_models = []
    
    # Generate all images
    for model_name in comparison.models.keys():
        if comparison.setup_model(model_name):
            successful_models.append(model_name)
            
            for scenario_name, scenario_data in comparison.scenarios.items():
                print(f"   🎨 {model_name} - {scenario_name}")
                
                image = comparison.generate_image(
                    scenario_data["prompt"], 
                    scenario_data["negative"]
                )
                
                if image:
                    output_path = os.path.join(output_dir, f"{scenario_name}_{model_name}.png")
                    image.save(output_path, quality=95)
                    print(f"      ✅ Saved")
                else:
                    print(f"      ❌ Failed")
    
    # Create the comparison grid
    comparison.create_comparison_grid(output_dir)
    
    print(f"\n✅ Complete! Check '{output_dir}' folder for:")
    print("   • Individual images: {scenario}_{model}.png")
    print("   • Comparison grid: 4_scenario_comparison_grid.png")
    
    # Cleanup
    if comparison.current_pipeline:
        del comparison.current_pipeline
    if comparison.compel:
        del comparison.compel
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()