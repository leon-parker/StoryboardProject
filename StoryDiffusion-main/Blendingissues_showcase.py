"""
REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - NO MASKING VERSION
Demonstrates character blending issues without masking for research paper!

✅ KEY FEATURES:
- Uses Compel library to handle prompts longer than 77 tokens
- NO masking or ControlNet - shows raw character blending issues
- Detailed prompts for better SDXL results
- Simple IP-Adapter integration without spatial control
- Perfect for demonstrating WHY masking is necessary!
"""

import os
import torch
import time
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc

class CompelNoMaskingPipeline:
    """Pipeline with Compel support for long prompts - NO MASKING to show character blending"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.compel = None
        self.ip_adapter_loaded = False
        
    def setup_pipeline_with_compel(self):
        """Setup pipeline with Compel for long prompt support - NO ControlNet or masking"""
        print("🔧 Setting up RealCartoon XL v7 with Compel (NO MASKING - Character Blending Demo)...")
        
        try:
            # Verify model
            size_gb = os.path.getsize(self.model_path) / (1024**3)
            print(f"   📊 Model: {size_gb:.1f}GB")
            if 6.5 <= size_gb <= 7.5:
                print("   ✅ RealCartoon XL v7 confirmed")
            
            # Load image encoder first
            print("   📸 Loading image encoder...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
            )
            
            # Load ONLY regular pipeline - no ControlNet
            print("   🎯 Loading RealCartoon XL v7 (regular pipeline only)...")
            self.pipeline = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                image_encoder=image_encoder,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
            
            # Setup scheduler
            print("   📐 Setting up scheduler...")
            self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
            self.pipeline.enable_vae_tiling()
            
            # Setup Compel for long prompts (CRITICAL!)
            print("   🚀 Setting up Compel for long prompts...")
            self.compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False  # This enables long prompts!
            )
            
            print("✅ Pipeline with Compel ready!")
            print("   🎯 Long prompts (>77 tokens) now supported!")
            print("   ⚠️ NO MASKING - will demonstrate character blending issues!")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            traceback.print_exc()
            return False
    
    def load_ip_adapter_simple(self):
        """Load IP-Adapter with simple configuration"""
        print("\n🖼️ Loading IP-Adapter (NO masking support)...")
        
        try:
            # Load IP-Adapter for regular pipeline
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
            )
            self.ip_adapter_loaded = True
            print("   ✅ IP-Adapter loaded!")
            print("   ⚠️ NO masking capability - characters will blend!")
            
            return True
            
        except Exception as e:
            print(f"   ❌ IP-Adapter failed: {e}")
            return False
    
    def encode_prompts_with_compel(self, positive_prompt, negative_prompt):
        """Encode both prompts using Compel and ensure matching shapes"""
        try:
            print(f"   📝 Encoding positive prompt: {positive_prompt[:100]}{'...' if len(positive_prompt) > 100 else ''}")
            print(f"   📊 Positive prompt length: {len(positive_prompt.split())} words")
            
            print(f"   📝 Encoding negative prompt: {negative_prompt[:100]}{'...' if len(negative_prompt) > 100 else ''}")
            print(f"   📊 Negative prompt length: {len(negative_prompt.split())} words")
            
            # Use Compel to handle both prompts
            positive_conditioning, positive_pooled = self.compel(positive_prompt)
            negative_conditioning, negative_pooled = self.compel(negative_prompt)
            
            # CRITICAL: Pad conditioning tensors to same length to avoid shape mismatch
            print("   🔧 Padding tensors to same length...")
            [positive_conditioning, negative_conditioning] = self.compel.pad_conditioning_tensors_to_same_length(
                [positive_conditioning, negative_conditioning]
            )
            
            print(f"   ✅ Final tensor shapes: pos={positive_conditioning.shape}, neg={negative_conditioning.shape}")
            
            return positive_conditioning, positive_pooled, negative_conditioning, negative_pooled
            
        except Exception as e:
            print(f"   ❌ Compel encoding failed: {e}")
            traceback.print_exc()
            return None, None, None, None
    
    def generate_reference_with_compel(self, prompt, negative_prompt, **kwargs):
        """Generate character reference using Compel for long prompts"""
        try:
            # Encode both prompts with Compel and ensure matching shapes
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )
            
            if prompt_embeds is None:
                return None
            
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                **kwargs
            )
            
            # Clean up embeddings to save memory
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Reference generation failed: {e}")
            traceback.print_exc()
            return None
    
    def generate_single_character_scene_with_compel(self, prompt, negative_prompt, character_ref, **kwargs):
        """Generate single character scene using Compel"""
        
        if not self.ip_adapter_loaded:
            print("   ❌ IP-Adapter not loaded!")
            return None
        
        try:
            # Encode both prompts with Compel and ensure matching shapes
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )
            
            if prompt_embeds is None:
                return None
            
            # Simple IP-Adapter configuration
            self.pipeline.set_ip_adapter_scale(0.7)
            
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                ip_adapter_image=character_ref,
                **kwargs
            )
            
            # Clean up embeddings
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Single character generation failed: {e}")
            traceback.print_exc()
            return None
    
    def generate_dual_character_scene_NO_MASKING(self, prompt, negative_prompt, alex_ref, emma_ref, **kwargs):
        """Generate dual character scene WITHOUT masking - demonstrates blending issues!"""
        
        if not self.ip_adapter_loaded:
            print("   ❌ IP-Adapter not loaded!")
            return None
        
        try:
            print("   ⚠️ Generating dual characters WITHOUT masking - expect blending issues!")
            
            # Encode prompts with Compel
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )
            
            if prompt_embeds is None:
                return None
            
            # NO MASKING - just pass both character references directly
            # This will cause characters to blend/merge together!
            self.pipeline.set_ip_adapter_scale(0.7)
            
            # Try different approaches to show various blending issues:
            blending_method = kwargs.pop('blending_method', 'both_refs')
            
            if blending_method == 'both_refs':
                # Method 1: Pass both references - they will compete and blend
                print("   📝 Method: Both references (will compete and blend)")
                ip_adapter_image = [alex_ref, emma_ref]
                
            elif blending_method == 'single_dominant':
                # Method 2: Use one reference - other character gets lost
                print("   📝 Method: Single reference (one character dominates)")
                ip_adapter_image = alex_ref  # Emma will likely be lost or distorted
                
            elif blending_method == 'averaged':
                # Method 3: Try to blend references - creates hybrid features
                print("   📝 Method: Reference blending (creates hybrid features)")
                ip_adapter_image = [alex_ref, emma_ref]
                self.pipeline.set_ip_adapter_scale([0.5, 0.5])  # Equal weights
            
            # Generate WITHOUT any spatial control or masking
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                ip_adapter_image=ip_adapter_image,
                **kwargs
            )
            
            print("   ⚠️ Generated without masking - check for character blending!")
            
            # Clean up embeddings
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
            
            return result.images[0] if result and result.images else None
            
        except Exception as e:
            print(f"❌ Dual character generation failed: {e}")
            traceback.print_exc()
            return None

class DetailedEducationalPrompts:
    """Detailed prompts optimized for SDXL with Compel (no 77-token limit!)"""
    
    def __init__(self):
        # Detailed character descriptions for SDXL
        self.alex_base = """Alex, a cheerful 6-year-old boy with short neat black hair and warm expressive brown eyes, 
        wearing a comfortable bright blue t-shirt and dark comfortable pants, displaying a genuine friendly smile 
        and kind facial expression, with a natural child-like innocent demeanor"""
        
        self.emma_base = """Emma, a sweet 6-year-old girl with blonde hair styled in neat ponytails held by blue ribbons, 
        bright blue eyes full of curiosity, wearing a soft pink cardigan over a white shirt, 
        with a gentle sweet expression and delicate youthful features, displaying a thoughtful and sensitive nature"""
        
        self.setting_base = """bright cheerful classroom environment with warm natural lighting, 
        educational cartoon art style with soft colors and clean lines, child-friendly atmosphere, 
        professional educational illustration quality"""
        
    def get_reference_prompt(self, character_name):
        """Detailed reference prompts (no token limit with Compel!)"""
        if character_name == "Alex":
            return f"""{self.alex_base}, character reference sheet, full body portrait, 
            standing pose, clean white background, high quality digital illustration, 
            consistent character design, educational animation style, bright even lighting, 
            clear details, professional character concept art"""
        else:
            return f"""{self.emma_base}, character reference sheet, full body portrait, 
            standing pose, clean white background, high quality digital illustration, 
            consistent character design, educational animation style, bright even lighting, 
            clear details, professional character concept art"""
    
    def get_scene_prompt(self, skill, character_focus, action):
        """Detailed scene prompts (no token limit with Compel!)"""
        if character_focus == "Alex":
            return f"""{self.alex_base}, {action}, {self.setting_base}, 
            autism education themed illustration, social skills learning context, 
            high quality educational content, detailed facial expressions showing emotion, 
            natural body language and gestures, warm inviting atmosphere, 
            child-appropriate content, positive educational messaging"""
        elif character_focus == "Emma":
            return f"""{self.emma_base}, {action}, {self.setting_base}, 
            autism education themed illustration, social skills learning context, 
            high quality educational content, detailed facial expressions showing emotion, 
            natural body language and gestures, warm inviting atmosphere, 
            child-appropriate content, positive educational messaging"""
        else:  # both characters - this will show blending issues!
            return f"""Alex and Emma, two distinct children with clearly different appearances, Alex with black hair and blue shirt, Emma with blonde ponytails and pink cardigan, {action}, {self.setting_base}, 
            social skills education illustration, autism awareness content, showing positive social interaction, 
            detailed character expressions and body language, warm educational atmosphere, 
            high quality digital illustration, child-friendly educational content, 
            demonstrating friendship and social learning, bright cheerful colors, 
            two separate children clearly visible, distinct character features"""
    
    def get_negative_prompt(self):
        """Enhanced negative prompt to try to prevent blending (though it won't work without masking)"""
        return """realistic photography, photorealistic, adult faces, teenage appearance, 
        blurry image, low quality, poor resolution, dark lighting, scary atmosphere, 
        inappropriate content, weapons, violence, sad or distressing themes, 
        poor anatomy, distorted faces, multiple heads, extra limbs, 
        bad hands, mutation, deformed features, background characters, other people,
        merged faces, blended features, hybrid characters, face morphing, character fusion"""

def find_realcartoon_model():
    """Find RealCartoon XL v7 model"""
    paths = [
        "../models/realcartoonxl_v7.safetensors",
        "models/realcartoonxl_v7.safetensors",
        "../models/RealCartoon-XL-v7.safetensors"
    ]
    
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def main():
    """RealCartoon XL v7 with Compel Long Prompts - NO MASKING (Character Blending Demo)"""
    print("\n" + "="*70)
    print("🚀 REALCARTOON XL V7 - NO MASKING VERSION (CHARACTER BLENDING DEMO)")
    print("   Demonstrates WHY masking is necessary for dual character scenes!")
    print("="*70)
    
    print("\n🎯 Demo Features:")
    print("  • Uses Compel for long prompts (>77 tokens)")
    print("  • NO masking or spatial control")
    print("  • NO ControlNet")
    print("  • Shows character blending/merging issues")
    print("  • Perfect for research paper comparison!")
    print("  • Demonstrates the PROBLEM that masking solves")
    
    print("\n⚠️ EXPECTED ISSUES WITHOUT MASKING:")
    print("  • Characters will blend together")
    print("  • One character may dominate")
    print("  • Hybrid facial features")
    print("  • Inconsistent character appearance")
    print("  • Poor spatial separation")
    
    # Check if Compel is installed
    try:
        import compel
        print("  ✅ Compel library detected")
    except ImportError:
        print("  ❌ Compel library not found!")
        print("     Install with: pip install compel")
        return
    
    # Find model
    model_path = find_realcartoon_model()
    if not model_path:
        model_path = input("📁 Enter RealCartoon XL v7 path: ").strip()
        if not os.path.exists(model_path):
            print("❌ Model not found")
            return
    
    output_dir = "no_masking_blending_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup pipeline with Compel (NO masking/ControlNet)
    pipeline = CompelNoMaskingPipeline(model_path)
    if not pipeline.setup_pipeline_with_compel():
        print("❌ Pipeline setup failed")
        return
    
    prompts = DetailedEducationalPrompts()
    
    # Define comprehensive story scenes to show blending issues
    blending_demo_scenes = [
        # Single character scenes for reference
        ("01_alex_noticing", "single", "Alex", 
         "Alex alone, looking around the classroom with a concerned and empathetic expression on his face, with gentle body language showing he wants to help someone, kind and thoughtful demeanor"),
        
        ("02_emma_lonely", "single", "Emma", 
         "Emma alone, sitting at a desk with a book, looking sad and wistful, hoping for friendship and social connection, with expressive eyes showing her desire to interact with others"),
        
        ("03_alex_thinking", "single", "Alex", 
         "Alex alone at his desk with a thinking expression, pondering how to approach and help Emma"),
        
        # Dual character scenes WITHOUT masking - will show blending issues
        ("04_approaching", "dual", "both", 
         "Alex walking towards Emma at her desk while Emma looks up with cautious but hopeful expression, showing the beginning of positive social interaction",
         "both_refs"),
        
        ("05_greeting", "dual", "both", 
         "Alex and Emma doing a high five together, hands raised up and touching, both smiling happily, celebrating their new friendship",
         "both_refs"),
        
        ("06_playing_together", "dual", "both", 
         "Alex and Emma playing with educational toys together, showing cooperation and shared enjoyment, with happy expressions and collaborative interaction",
         "both_refs"),
        
        ("07_sharing", "dual", "both", 
         "Alex and Emma sharing books and educational materials, demonstrating generosity and friendship, with caring expressions and positive social behavior",
         "both_refs"),
        
        ("08_conversation", "dual", "both", 
         "Alex and Emma having a friendly conversation, sitting at a table facing each other, both engaged and listening to each other",
         "single_dominant"),
        
        ("09_learning_together", "dual", "both", 
         "Alex and Emma working on a puzzle together, heads close together as they collaborate, showing teamwork and friendship",
         "high_conflict"),
        
        ("10_comfort", "dual", "both", 
         "Alex gently comforting Emma, placing a reassuring hand on her shoulder, showing empathy and kindness",
         "high_conflict"),
        
        ("11_celebration", "dual", "both", 
         "Alex and Emma jumping with joy and excitement, arms raised in celebration, both children clearly visible and happy",
         "high_conflict"),
        
        ("12_reading_together", "dual", "both", 
         "Alex and Emma sitting side by side reading the same book, heads tilted toward each other, sharing a story",
         "single_dominant"),
        
        ("13_art_project", "dual", "both", 
         "Alex and Emma working on an art project together, both holding crayons and drawing on the same paper",
         "both_refs"),
        
        ("14_lunch_sharing", "dual", "both", 
         "Alex and Emma sharing lunch at a table, Alex offering part of his sandwich to Emma, showing generosity",
         "both_refs"),
        
        ("15_playground_fun", "dual", "both", 
         "Alex and Emma playing on playground equipment together, both laughing and having fun, showing their friendship",
         "averaged"),
    ]
    
    try:
        # PHASE 1: Character references with long prompts
        print("\n👦👧 PHASE 1: Character References (For Comparison)")
        print("-" * 60)
        
        print("🎨 Generating Alex reference...")
        alex_prompt = prompts.get_reference_prompt("Alex")
        negative = prompts.get_negative_prompt()
        
        print(f"   📊 Alex prompt length: {len(alex_prompt.split())} words")
        
        alex_ref = pipeline.generate_reference_with_compel(
            alex_prompt, negative,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(1000)
        )
        
        print("🎨 Generating Emma reference...")
        emma_prompt = prompts.get_reference_prompt("Emma")
        
        print(f"   📊 Emma prompt length: {len(emma_prompt.split())} words")
        
        emma_ref = pipeline.generate_reference_with_compel(
            emma_prompt, negative,
            num_inference_steps=30,
            guidance_scale=7.5,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(2000)
        )
        
        if not alex_ref or not emma_ref:
            print("❌ Character reference generation failed")
            return
        
        alex_ref.save(os.path.join(output_dir, "alex_reference.png"))
        emma_ref.save(os.path.join(output_dir, "emma_reference.png"))
        print("✅ Character references saved")
        
        # PHASE 2: Load IP-Adapter
        print("\n🔄 Loading IP-Adapter (NO masking support)...")
        if not pipeline.load_ip_adapter_simple():
            print("❌ IP-Adapter loading failed")
            return
        
        # PHASE 3: Generate scenes to demonstrate blending issues
        print(f"\n🎬 PHASE 3: Blending Demo Scenes ({len(blending_demo_scenes)} total)")
        print("-" * 70)
        
        results = []
        total_time = 0
        blending_issues = []
        
        for i, scene_data in enumerate(blending_demo_scenes):
            if len(scene_data) == 5:
                scene_name, scene_type, character_focus, action, blending_method = scene_data
            else:
                scene_name, scene_type, character_focus, action = scene_data
                blending_method = "both_refs"
            
            print(f"\n🎨 Scene {i+1}: {scene_name} ({scene_type})")
            if scene_type == "dual":
                print(f"   ⚠️ Blending method: {blending_method} - expect issues!")
            
            prompt = prompts.get_scene_prompt("social skills", character_focus, action)
            output_path = os.path.join(output_dir, f"{scene_name}.png")
            
            print(f"   📊 Scene prompt length: {len(prompt.split())} words")
            
            start_time = time.time()
            
            # Use appropriate generation method
            if scene_type == "single":
                character_ref = alex_ref if character_focus == "Alex" else emma_ref
                image = pipeline.generate_single_character_scene_with_compel(
                    prompt, negative, character_ref,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(1000 + i)
                )
            else:  # dual - will show blending issues
                image = pipeline.generate_dual_character_scene_NO_MASKING(
                    prompt, negative, alex_ref, emma_ref,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(1000 + i),
                    blending_method=blending_method
                )
                blending_issues.append(scene_name)
            
            scene_time = time.time() - start_time
            total_time += scene_time
            
            if image:
                image.save(output_path, quality=95, optimize=True)
                results.append(scene_name)
                print(f"   ✅ Generated: {scene_name} ({scene_time:.1f}s)")
                if scene_type == "dual":
                    print(f"   📝 Check for blending issues in: {scene_name}")
            else:
                print(f"   ❌ Failed: {scene_name}")
        
        # Results
        print("\n" + "=" * 70)
        print("🚀 NO MASKING DEMO RESULTS - CHARACTER BLENDING SHOWCASE")
        print("=" * 70)
        
        success_rate = len(results) / len(blending_demo_scenes) * 100
        avg_time = total_time / len(blending_demo_scenes) if blending_demo_scenes else 0
        
        print(f"✓ Success rate: {success_rate:.1f}% ({len(results)}/{len(blending_demo_scenes)})")
        print(f"✓ Average time: {avg_time:.1f}s per scene")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Used Compel for long prompts")
        print(f"⚠️ NO masking used - expect character blending!")
        
        if results:
            print("\n📋 Generated scenes:")
            for result in results:
                print(f"   • {result}")
        
        if blending_issues:
            print("\n⚠️ SCENES WITH EXPECTED BLENDING ISSUES:")
            for issue in blending_issues:
                print(f"   • {issue} - check for character merging/blending")
        
        print("\n🔬 RESEARCH PAPER ANALYSIS GUIDE:")
        print("   📊 WHAT TO LOOK FOR IN DUAL CHARACTER SCENES:")
        print("     • Character feature mixing (Alex's hair + Emma's face)")
        print("     • One character completely missing")
        print("     • Hybrid clothing (blue-pink blended shirts)")
        print("     • Inconsistent character positioning")
        print("     • Facial feature averaging/morphing")
        print("     • Hair color blending (brown-blonde mix)")
        print("     • Size/proportion inconsistencies")
        print("     • Clothing pattern conflicts")
        
        print("\n   📈 COMPARISON METRICS FOR YOUR PAPER:")
        print("     • Character recognition accuracy")
        print("     • Feature consistency across scenes")
        print("     • Spatial separation quality")
        print("     • Individual character preservation")
        print("     • Color palette integrity")
        
        print("\n   📝 DOCUMENTATION SUGGESTIONS:")
        print("     • Compare single vs dual scene quality scores")
        print("     • Document specific blending artifacts")
        print("     • Measure character feature preservation")
        print("     • Analyze spatial distribution problems")
        print("     • Quantify the improvement with masking")
        
        print("\n🎉 DEMO ADVANTAGES FOR RESEARCH:")
        print("   • Clear demonstration of the core problem")
        print("   • Multiple failure modes documented")
        print("   • Provides quantitative comparison baseline")
        print("   • Shows why masking is technically necessary")
        print("   • Validates your solution's importance")
        print("   • Creates compelling before/after narrative")
        
        if success_rate >= 70:
            print("\n🚀 COMPREHENSIVE BLENDING DEMO SUCCESS!")
            print("   📚 Complete story sequence with blending issues")
            print("   🎭 Multiple failure modes demonstrated")
            print("   🎨 Perfect for research paper methodology section")
            print("   ✨ Shows the problem your solution addresses!")
            print("   📊 Ready for quantitative analysis!")
        
        print(f"\n📁 Outputs: {os.path.abspath(output_dir)}/")
        print("\n📝 FOR YOUR RESEARCH PAPER:")
        print("   📋 METHODOLOGY SECTION:")
        print("     • Use single character scenes as baseline quality")
        print("     • Document dual character failure modes")
        print("     • Show progression of story with/without masking")
        print("   📊 RESULTS SECTION:")
        print("     • Quantify character preservation metrics")
        print("     • Compare spatial consistency scores")
        print("     • Measure feature blending artifacts")
        print("   📈 DISCUSSION SECTION:")
        print("     • Explain why masking solves these issues")
        print("     • Demonstrate technical necessity of your approach")
        print("     • Show improvement in narrative coherence")
        print("   🖼️ FIGURES:")
        print("     • Before/after comparison grids")
        print("     • Character consistency analysis")
        print("     • Story progression visualization")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        if pipeline.pipeline:
            del pipeline.pipeline
        if pipeline.compel:
            del pipeline.compel
        torch.cuda.empty_cache()
        print("✅ Complete")


if __name__ == "__main__":
    main()