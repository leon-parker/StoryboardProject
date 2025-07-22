"""

REALCARTOON XL V7 WITH COMPEL LONG PROMPTS - ADULT COFFEE SHOP SOCIAL STORY

Solves the 77-token CLIP limit using the Compel library!

✅ KEY FEATURES:

- Uses Compel library to handle prompts longer than 77 tokens
- Automatic prompt chunking and concatenation
- Support for SDXL dual text encoders
- Detailed prompts for better SDXL results
- Simple IP-Adapter integration
- Adult autism social story: "Ordering Coffee with Confidence"
- Follows Carol Gray's Social Stories™ 10.3 criteria

"""

import os
import torch
import time
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from compel import Compel, ReturnedEmbeddingsType
import traceback
import gc
import numpy as np

class CompelRealCartoonPipeline:
    """Pipeline with Compel support for long prompts - optimized for adult social stories"""

    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.compel = None
        self.ip_adapter_loaded = False
       
    def setup_pipeline_with_compel(self):
        """Setup pipeline with Compel for long prompt support"""
        print("🔧 Setting up RealCartoon XL v7 with Compel (Long Prompts)...")
       
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
           
            # Load regular pipeline
            print("   🎯 Loading RealCartoon XL v7...")
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
            return True
           
        except Exception as e:
            print(f"❌ Pipeline setup failed: {e}")
            traceback.print_exc()
            return False
   
    def load_ip_adapter_simple(self):
        """Load IP-Adapter with simple configuration"""
        print("\n🖼️ Loading IP-Adapter...")
       
        try:
            # Load IP-Adapter for pipeline
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"]
            )
            self.ip_adapter_loaded = True
            print("   ✅ IP-Adapter loaded!")
           
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
            with torch.no_grad():  # Memory efficiency
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
   
    def create_simple_masks(self, width=1024, height=1024):
        """Create simple left/right masks for dual character scenes"""
        # Jamie mask - left side
        jamie_mask = Image.new("L", (width, height), 0)
        jamie_draw = ImageDraw.Draw(jamie_mask)
        jamie_draw.rectangle([0, 0, width//2 + 100, height], fill=255)
       
        # Casey mask - right side  
        casey_mask = Image.new("L", (width, height), 0)
        casey_draw = ImageDraw.Draw(casey_mask)
        casey_draw.rectangle([width//2 - 100, 0, width, height], fill=255)
       
        # Light blur for smooth edges
        jamie_mask = jamie_mask.filter(ImageFilter.GaussianBlur(radius=2))
        casey_mask = casey_mask.filter(ImageFilter.GaussianBlur(radius=2))
       
        return jamie_mask, casey_mask
   
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
            self.pipeline.set_ip_adapter_scale(0.75)
           
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
   
    def generate_dual_character_scene_with_compel(self, prompt, negative_prompt, jamie_ref, casey_ref, **kwargs):
        """Generate dual character scene using Compel and IP-Adapter masking"""
       
        if not self.ip_adapter_loaded:
            print("   ❌ IP-Adapter not loaded!")
            return None
       
        try:
            print("   📝 Using regular pipeline for dual character scene...")
           
            # Encode prompts with Compel
            prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled = self.encode_prompts_with_compel(
                prompt, negative_prompt
            )
           
            if prompt_embeds is None:
                return None
           
            # Create simple left/right masks
            jamie_mask, casey_mask = self.create_simple_masks()
           
            # Process masks
            processor = IPAdapterMaskProcessor()
            masks = processor.preprocess([jamie_mask, casey_mask], height=1024, width=1024)
           
            # IP-Adapter configuration for masking
            self.pipeline.set_ip_adapter_scale([[0.75, 0.75]])
           
            # Format for masking
            ip_images = [[jamie_ref, casey_ref]]
            masks_formatted = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]
           
            # Use regular pipeline
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled,
                ip_adapter_image=ip_images,
                cross_attention_kwargs={"ip_adapter_masks": masks_formatted},
                **kwargs
            )
           
            # Clean up embeddings
            del prompt_embeds, pooled_prompt_embeds, negative_embeds, negative_pooled
            gc.collect()
           
            return result.images[0] if result and result.images else None
           
        except Exception as e:
            print(f"❌ Dual character generation failed: {e}")
            traceback.print_exc()
            return None

class AdultCoffeeShopSocialStoryPrompts:
    """Adult autism social story prompts following Carol Gray's Social Stories™ criteria"""
   
    def __init__(self):
        # Detailed adult character descriptions for SDXL
        self.jamie_base = """Jamie, a 26-year-old young adult with autism, neat shoulder-length brown hair and thoughtful hazel eyes,
        wearing a comfortable navy blue sweater and dark jeans, displaying a gentle but slightly nervous expression,
        carrying a small backpack, with a natural adult appearance and determined demeanor despite social anxiety"""
       
        self.casey_base = """Casey, a friendly 24-year-old barista with warm brown skin and curly black hair in a neat bun,
        wearing a clean green apron over a white t-shirt, bright welcoming smile and patient understanding eyes,
        with professional but approachable body language, standing behind a coffee shop counter with confidence"""
       
        self.setting_base = """modern bright coffee shop interior with warm lighting, clean wooden counter and menu boards,
        coffee machines and pastry display cases, cozy atmosphere with soft background music,
        professional service environment, adult-friendly space with comfortable seating areas"""
       
    def get_reference_prompt(self, character_name):
        """Detailed character reference prompts for adults"""
        if character_name == "Jamie":
            return f"""{self.jamie_base}, character reference sheet, full body portrait,
            standing pose in casual professional attire, clean white background, high quality digital illustration,
            consistent adult character design, realistic cartoon art style, bright even lighting,
            clear facial features and expressions, professional character concept art for social story"""
        else:
            return f"""{self.casey_base}, character reference sheet, full body portrait,
            standing pose in work uniform, clean white background, high quality digital illustration,
            consistent adult character design, realistic cartoon art style, bright even lighting,
            clear facial features and expressions, professional character concept art for social story"""
   
    def get_scene_prompt(self, character_focus, action):
        """Detailed scene prompts following Social Stories™ criteria"""
        if character_focus == "Jamie":
            return f"""{self.jamie_base}, {action}, {self.setting_base},
            adult autism social story illustration, anxiety management and social skills context,
            high quality educational content for adults, detailed facial expressions showing realistic emotions,
            natural adult body language and gestures, supportive learning environment,
            adult-appropriate content, positive educational messaging about social confidence,
            professional illustration quality suitable for adult learners"""
        elif character_focus == "Casey":
            return f"""{self.casey_base}, {action}, {self.setting_base},
            adult autism social story illustration, customer service and patience context,
            high quality educational content for adults, detailed facial expressions showing understanding,
            natural adult body language and gestures, supportive learning environment,
            adult-appropriate content, positive educational messaging about helpful service,
            professional illustration quality suitable for adult learners"""
        else:  # both characters
            return f"""Jamie and Casey, two adults interacting in a coffee shop setting, {action}, {self.setting_base},
            adult autism social story illustration, social skills and confidence building context,
            showing positive customer service interaction between adult with autism and understanding barista,
            detailed character expressions and professional body language, supportive learning environment,
            high quality digital illustration, adult-appropriate educational content,
            demonstrating successful social interaction and confidence building, realistic cartoon art style"""
   
    def get_negative_prompt(self):
        """Negative prompt optimized for adult social stories"""
        return """child-like appearance, teenage faces, immature features, cartoon exaggeration,
        unrealistic proportions, blurry image, low quality, poor resolution, dark lighting, scary atmosphere,
        inappropriate content, weapons, violence, distressing themes, unprofessional appearance,
        poor anatomy, distorted faces, multiple heads, extra limbs, bad hands, mutation, deformed features,
        background crowds, other people, distracting elements, overly stylized, anime style"""

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
    """RealCartoon XL v7 with Compel - Adult Coffee Shop Social Story"""
    print("\n" + "="*80)
    print("☕ REALCARTOON XL V7 - ADULT COFFEE SHOP SOCIAL STORY")
    print("   'Ordering Coffee with Confidence' - Following Carol Gray's Social Stories™")
    print("="*80)
   
    print("\n🎯 Social Story Features:")
    print("  • Adult autism social story: 'Ordering Coffee with Confidence'")
    print("  • Follows Carol Gray's Social Stories™ 10.3 criteria")
    print("  • Meaningful, safe, patient, and non-judgmental content")
    print("  • Realistic adult characters and situations")
    print("  • Step-by-step social interaction guidance")
   
    print("\n🚀 Technical Features:")
    print("  • Handles prompts longer than 77 tokens")
    print("  • Automatic prompt chunking and concatenation")
    print("  • SDXL dual text encoder support")
    print("  • Memory-efficient implementation")
    print("  • No more 'truncated prompt' warnings!")
   
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
   
    output_dir = "coffee_shop_social_story"
    os.makedirs(output_dir, exist_ok=True)
   
    # Setup pipeline with Compel
    pipeline = CompelRealCartoonPipeline(model_path)
    if not pipeline.setup_pipeline_with_compel():
        print("❌ Pipeline setup failed")
        return
   
    prompts = AdultCoffeeShopSocialStoryPrompts()
   
    # Define the Coffee Shop Social Story scenes
    coffee_shop_scenes = [
        # Character introductions
        ("01_jamie_outside", "single", "Jamie",
         "Jamie standing outside the coffee shop, looking at the entrance with a mix of determination and nervousness, taking a deep breath before entering, showing courage despite social anxiety"),
       
        ("02_casey_ready", "single", "Casey",
         "Casey standing behind the coffee shop counter, wiping down surfaces and organizing cups, smiling warmly and ready to help customers, displaying professional and patient demeanor"),
       
        # Social story progression
        ("03_jamie_enters", "single", "Jamie",
         "Jamie entering the coffee shop, looking around the welcoming interior, noticing the menu board and counter, appearing slightly overwhelmed but determined to order"),
       
        ("04_casey_greets", "dual", "both",
         "Casey greeting Jamie with a warm smile and friendly wave, saying 'Good morning! What can I get for you today?' while Jamie approaches the counter looking nervous but hopeful"),
       
        ("05_jamie_looking_menu", "single", "Jamie",
         "Jamie standing at the counter, looking up at the menu board with a concentrated expression, feeling slightly overwhelmed by the choices, taking time to decide"),
       
        ("06_casey_offers_help", "dual", "both",
         "Casey noticing Jamie's hesitation and offering help with a patient smile, saying 'Take your time, or would you like me to recommend something popular?' while Jamie looks relieved"),
       
        ("07_jamie_orders", "dual", "both",
         "Jamie pointing to the menu and saying 'I'd like a medium coffee, please' with growing confidence, while Casey nods encouragingly and reaches for a cup"),
       
        ("08_casey_confirms", "dual", "both",
         "Casey confirming the order with a friendly smile, saying 'One medium coffee coming right up! That'll be $3.50' while Jamie nods and reaches for payment"),
       
        ("09_jamie_pays", "dual", "both",
         "Jamie carefully counting out exact change or using a card to pay, feeling proud of managing the transaction, while Casey patiently waits and smiles supportively"),
       
        ("10_casey_makes_coffee", "single", "Casey",
         "Casey preparing Jamie's coffee with professional skill, operating the coffee machine while making friendly small talk, creating a comfortable atmosphere"),
       
        ("11_transaction_complete", "dual", "both",
         "Casey handing Jamie the coffee with a warm smile, saying 'Here you go! Have a great day!' while Jamie accepts the coffee with a genuine smile of accomplishment"),
       
        ("12_jamie_confident", "single", "Jamie",
         "Jamie walking away from the counter with coffee in hand, displaying a proud and confident expression, feeling accomplished after successfully completing the social interaction"),
    ]
   
    try:
        # PHASE 1: Character references
        print("\n👥 PHASE 1: Creating Adult Character References")
        print("-" * 60)
       
        print("☕ Generating Jamie (customer) reference...")
        jamie_prompt = prompts.get_reference_prompt("Jamie")
        negative = prompts.get_negative_prompt()
       
        print(f"   📊 Jamie prompt length: {len(jamie_prompt.split())} words")
       
        jamie_ref = pipeline.generate_reference_with_compel(
            jamie_prompt, negative,
            num_inference_steps=35,
            guidance_scale=8.0,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(2024)
        )
       
        print("👩‍💼 Generating Casey (barista) reference...")
        casey_prompt = prompts.get_reference_prompt("Casey")
       
        print(f"   📊 Casey prompt length: {len(casey_prompt.split())} words")
       
        casey_ref = pipeline.generate_reference_with_compel(
            casey_prompt, negative,
            num_inference_steps=35,
            guidance_scale=8.0,
            height=1024, width=1024,
            generator=torch.Generator(pipeline.device).manual_seed(2025)
        )
       
        if not jamie_ref or not casey_ref:
            print("❌ Character reference generation failed")
            return
       
        jamie_ref.save(os.path.join(output_dir, "jamie_reference.png"))
        casey_ref.save(os.path.join(output_dir, "casey_reference.png"))
        print("✅ Adult character references saved")
       
        # PHASE 2: Load IP-Adapter
        print("\n🔄 Loading IP-Adapter...")
        if not pipeline.load_ip_adapter_simple():
            print("❌ IP-Adapter loading failed")
            return
       
        # PHASE 3: Generate social story scenes
        print(f"\n☕ PHASE 3: Coffee Shop Social Story ({len(coffee_shop_scenes)} scenes)")
        print("-" * 80)
       
        results = []
        total_time = 0
       
        for i, (scene_name, scene_type, character_focus, action) in enumerate(coffee_shop_scenes):
            print(f"\n🎬 Scene {i+1}: {scene_name} ({scene_type})")
           
            prompt = prompts.get_scene_prompt(character_focus, action)
            output_path = os.path.join(output_dir, f"{scene_name}.png")
           
            print(f"   📊 Scene prompt length: {len(prompt.split())} words")
           
            start_time = time.time()
           
            # Use appropriate generation method
            if scene_type == "single":
                character_ref = jamie_ref if character_focus == "Jamie" else casey_ref
                image = pipeline.generate_single_character_scene_with_compel(
                    prompt, negative, character_ref,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(3000 + i)
                )
            else:  # dual
                image = pipeline.generate_dual_character_scene_with_compel(
                    prompt, negative, jamie_ref, casey_ref,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=1024, width=1024,
                    generator=torch.Generator(pipeline.device).manual_seed(3000 + i)
                )
           
            scene_time = time.time() - start_time
            total_time += scene_time
           
            if image:
                image.save(output_path, quality=95, optimize=True)
                results.append(scene_name)
                print(f"   ✅ Generated: {scene_name} ({scene_time:.1f}s)")
            else:
                print(f"   ❌ Failed: {scene_name}")
       
        # Results
        print("\n" + "=" * 80)
        print("☕ COFFEE SHOP SOCIAL STORY RESULTS")
        print("=" * 80)
       
        success_rate = len(results) / len(coffee_shop_scenes) * 100
        avg_time = total_time / len(coffee_shop_scenes) if coffee_shop_scenes else 0
       
        print(f"✓ Success rate: {success_rate:.1f}% ({len(results)}/{len(coffee_shop_scenes)})")
        print(f"✓ Average time: {avg_time:.1f}s per scene")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Used Compel for long prompts")
        print(f"✓ No 77-token truncation warnings!")
        print(f"✓ Adult-appropriate social story content")
       
        if results:
            print(f"\n📋 Generated scenes for 'Ordering Coffee with Confidence':")
            for i, result in enumerate(results, 1):
                print(f"   {i:2d}. {result.replace('_', ' ').title()}")
       
        print(f"\n🎉 SOCIAL STORY SUCCESS!")
        print("   📚 Adult autism social story: 'Ordering Coffee with Confidence'")
        print("   📖 Following Carol Gray's Social Stories™ criteria")
        print("   🎭 Realistic adult character consistency")
        print("   🎨 Long detailed prompts working perfectly")
        print("   ✨ 77-token CLIP limit solved!")
        print("   🌟 Step-by-step social interaction guidance")
       
        if success_rate >= 80:
            print(f"\n🚀 READY FOR USE!")
            print("   🎯 Perfect for adult autism social skills training")
            print("   📝 Can be used by therapists, educators, or self-advocates")
            print("   💪 Builds confidence for real-world social situations")
            print("   🔄 Scenes can be reviewed as needed for reinforcement")
       
        print(f"\n📁 Complete social story saved to: {os.path.abspath(output_dir)}/")
        print("   📖 Use these images in sequence to tell the complete story")
        print("   🎯 Focus on one scene at a time for best learning outcomes")
       
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
        print("\n✅ Generation complete!")

if __name__ == "__main__":
    main()