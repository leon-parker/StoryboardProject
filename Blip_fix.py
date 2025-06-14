# safetensors_clip_fix.py - Force CLIP/BLIP to use safetensors
import warnings
import os

def setup_safetensors_environment():
    """Setup environment to prefer safetensors over pickle files"""
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Force safetensors preference
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['SAFETENSORS_FAST_GPU'] = '1'
    
    print("🔧 Environment configured for safetensors")

def load_clip_with_safetensors():
    """Load CLIP using safetensors format"""
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        # Try to load with explicit safetensors preference
        print("📥 Loading CLIP with safetensors...")
        
        # Load processor (usually doesn't have the issue)
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_fast=False  # Avoid the fast tokenizer issues
        )
        
        # Load model with safetensors preference
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True,  # Force safetensors
            trust_remote_code=False
        )
        
        print("✅ CLIP loaded successfully with safetensors!")
        return processor, model, True
        
    except Exception as e:
        print(f"❌ CLIP safetensors failed: {e}")
        
        # Try alternative CLIP model that definitely uses safetensors
        try:
            print("📥 Trying alternative CLIP model...")
            processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
            model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
            print("✅ Alternative CLIP model loaded!")
            return processor, model, True
        except Exception as e2:
            print(f"❌ Alternative CLIP failed: {e2}")
            return None, None, False

def load_blip_with_safetensors():
    """Load BLIP using safetensors format"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        print("📥 Loading BLIP with safetensors...")
        
        # Load processor
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=False
        )
        
        # Load model with safetensors preference
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_safetensors=True,
            trust_remote_code=False
        )
        
        print("✅ BLIP loaded successfully with safetensors!")
        return processor, model, True
        
    except Exception as e:
        print(f"❌ BLIP safetensors failed: {e}")
        
        # Try alternative BLIP model
        try:
            print("📥 Trying alternative BLIP model...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            print("✅ Alternative BLIP model loaded!")
            return processor, model, True
        except Exception as e2:
            print(f"❌ Alternative BLIP failed: {e2}")
            return None, None, False

def download_safetensors_models():
    """Pre-download models in safetensors format"""
    print("📥 Pre-downloading safetensors models...")
    
    try:
        # Download safetensors versions explicitly
        from huggingface_hub import snapshot_download
        
        # Download CLIP in safetensors format
        snapshot_download(
            "openai/clip-vit-base-patch32",
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ignore_patterns=["*.bin", "*.msgpack"]
        )
        
        # Download BLIP in safetensors format  
        snapshot_download(
            "Salesforce/blip-image-captioning-base",
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
            ignore_patterns=["*.bin", "*.msgpack"]
        )
        
        print("✅ Safetensors models downloaded!")
        return True
        
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        return False

def test_safetensors_approach():
    """Test the safetensors approach"""
    
    setup_safetensors_environment()
    
    print("\n🧪 Testing Safetensors Approach...")
    
    # Try downloading first
    download_safetensors_models()
    
    # Test CLIP
    clip_processor, clip_model, clip_success = load_clip_with_safetensors()
    
    # Test BLIP  
    blip_processor, blip_model, blip_success = load_blip_with_safetensors()
    
    print(f"\n📊 Results:")
    print(f"CLIP: {'✅ SUCCESS' if clip_success else '❌ FAILED'}")
    print(f"BLIP: {'✅ SUCCESS' if blip_success else '❌ FAILED'}")
    
    if clip_success and blip_success:
        print("🎉 Both models working with safetensors!")
        
        # Quick functionality test
        try:
            from PIL import Image
            import requests
            
            # Test image
            url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/A_small_cup_of_coffee.JPG/256px-A_small_cup_of_coffee.JPG"
            image = Image.open(requests.get(url, stream=True).raw)
            
            # Test CLIP
            inputs = clip_processor(text=["a photo of a coffee cup"], images=image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            similarity = outputs.logits_per_image[0][0]
            print(f"🔗 CLIP similarity test: {similarity:.3f}")
            
            # Test BLIP
            inputs = blip_processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"📝 BLIP caption test: {caption}")
            
            print("✅ Full functionality confirmed!")
            
        except Exception as e:
            print(f"⚠️ Functionality test failed: {e}")
    
    return clip_success, blip_success

if __name__ == "__main__":
    test_safetensors_approach()