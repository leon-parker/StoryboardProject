# proper_diffusers_storydiffusion.py
"""
Proper StoryDiffusion Integration with Diffusers
Based on the official repository and diffusers compatibility
Let's find the REAL diffusers way to use StoryDiffusion
"""

import sys
import os
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

def find_real_storydiffusion_integration():
    """
    Let's examine what's actually in the StoryDiffusion repository
    and find the proper diffusers integration
    """
    print("🔍 FINDING REAL STORYDIFFUSION DIFFUSERS INTEGRATION")
    print("=" * 60)
    
    # Check if we're in the StoryDiffusion directory
    files_to_examine = [
        "storydiffusionpipeline.py",  # This looks promising!
        "predict.py",
        "app.py",
        "gradio_app_sdxl_specific_id_low_vram.py",
        "utils/pipeline.py",
        "requirements.txt"
    ]
    
    found_files = []
    for file in files_to_examine:
        if os.path.exists(file):
            found_files.append(file)
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    # Let's examine storydiffusionpipeline.py - this sounds like the main diffusers integration!
    if "storydiffusionpipeline.py" in found_files:
        print(f"\n📄 Examining storydiffusionpipeline.py - this should be the diffusers integration!")
        try:
            with open("storydiffusionpipeline.py", 'r') as f:
                content = f.read()
            
            # Look for key diffusers patterns
            diffusers_patterns = [
                "from diffusers import",
                "StableDiffusionXLPipeline", 
                "class StoryDiffusionPipeline",
                "DiffusionPipeline",
                "__call__",
                "def generate"
            ]
            
            print(f"🔍 Looking for diffusers integration patterns:")
            for pattern in diffusers_patterns:
                if pattern in content:
                    print(f"✅ Contains: {pattern}")
                else:
                    print(f"❌ Missing: {pattern}")
            
            # Show the first 50 lines to understand the structure
            lines = content.split('\n')
            print(f"\n📖 First 50 lines of storydiffusionpipeline.py:")
            for i, line in enumerate(lines[:50]):
                print(f"{i+1:2d}: {line}")
                
        except Exception as e:
            print(f"❌ Error reading storydiffusionpipeline.py: {e}")
    
    # Check predict.py - this might show usage
    if "predict.py" in found_files:
        print(f"\n📄 Examining predict.py for usage patterns:")
        try:
            with open("predict.py", 'r') as f:
                content = f.read()
            
            # Look for how they initialize the pipeline
            usage_patterns = [
                "StoryDiffusionPipeline",
                "from_pretrained",
                "load_model",
                "__call__",
                "generate"
            ]
            
            for pattern in usage_patterns:
                if pattern in content:
                    print(f"✅ predict.py uses: {pattern}")
                    # Show context around the pattern
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern in line:
                            start = max(0, i-2)
                            end = min(len(lines), i+3)
                            print(f"   Context:")
                            for j in range(start, end):
                                prefix = ">>> " if j == i else "    "
                                print(f"   {prefix}{lines[j]}")
                            break
                            
        except Exception as e:
            print(f"❌ Error reading predict.py: {e}")
    
    return found_files

def try_proper_storydiffusion_import():
    """
    Try to import StoryDiffusion the proper way
    """
    print(f"\n🔌 TRYING PROPER STORYDIFFUSION IMPORT")
    print("=" * 60)
    
    # Try different import methods
    import_attempts = [
        # Method 1: Direct from storydiffusionpipeline
        ("from storydiffusionpipeline import StoryDiffusionPipeline", "storydiffusionpipeline.StoryDiffusionPipeline"),
        
        # Method 2: From diffusers (if it's registered)
        ("from diffusers import StoryDiffusionPipeline", "diffusers.StoryDiffusionPipeline"),
        
        # Method 3: From utils
        ("from utils.pipeline import StoryDiffusionPipeline", "utils.pipeline.StoryDiffusionPipeline"),
        
        # Method 4: Import the modules we know exist
        ("from utils.gradio_utils import SpatialAttnProcessor2_0", "gradio_utils.SpatialAttnProcessor2_0"),
    ]
    
    successful_imports = []
    
    for import_statement, description in import_attempts:
        try:
            print(f"🔄 Trying: {import_statement}")
            exec(import_statement)
            print(f"✅ SUCCESS: {description}")
            successful_imports.append((import_statement, description))
        except ImportError as e:
            print(f"❌ FAILED: {e}")
        except Exception as e:
            print(f"⚠️ ERROR: {e}")
    
    return successful_imports

def create_proper_diffusers_pipeline():
    """
    Create a proper diffusers pipeline with StoryDiffusion
    Based on what we find in the actual code
    """
    print(f"\n🚀 CREATING PROPER DIFFUSERS STORYDIFFUSION PIPELINE")
    print("=" * 60)
    
    # Find model
    model_paths = [
        "../models/realcartoonxl_v7.safetensors",
        "../../models/realcartoonxl_v7.safetensors",
        "../realcartoonxl_v7.safetensors"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Model not found")
        return None
    
    print(f"📁 Using model: {model_path}")
    
    # Try to create StoryDiffusion pipeline the proper way
    try:
        # Method 1: Try the dedicated StoryDiffusion pipeline
        print("🔄 Attempting Method 1: StoryDiffusion Pipeline")
        try:
            from storydiffusionpipeline import StoryDiffusionPipeline
            
            pipe = StoryDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            
            print("✅ SUCCESS: Created StoryDiffusionPipeline!")
            return pipe, "StoryDiffusionPipeline"
            
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Try regular SDXL + StoryDiffusion patching
        print("🔄 Attempting Method 2: SDXL + StoryDiffusion patching")
        try:
            # Load regular SDXL pipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            
            # Try to find and apply StoryDiffusion patching
            try:
                from utils.gradio_utils import SpatialAttnProcessor2_0
                
                # Apply StoryDiffusion processors
                attn_procs = {}
                for name in pipe.unet.attn_processors.keys():
                    if name.endswith("attn1.processor"):  # Self-attention only
                        attn_procs[name] = SpatialAttnProcessor2_0(
                            hidden_size=pipe.unet.config.block_out_channels[-1],
                            cross_attention_dim=None
                        )
                    else:
                        attn_procs[name] = pipe.unet.attn_processors[name]
                
                pipe.unet.set_attn_processor(attn_procs)
                print("✅ SUCCESS: Applied StoryDiffusion to SDXL Pipeline!")
                return pipe, "SDXL + StoryDiffusion"
                
            except Exception as e:
                print(f"❌ StoryDiffusion patching failed: {e}")
                return pipe, "Plain SDXL (no StoryDiffusion)"
                
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
    
    except Exception as e:
        print(f"❌ All methods failed: {e}")
        return None, None

def main():
    """
    Main function to find and test proper StoryDiffusion integration
    """
    print("🚀 PROPER DIFFUSERS STORYDIFFUSION INTEGRATION")
    print("📄 Let's find how StoryDiffusion REALLY works with diffusers")
    print("🎯 No more hacks - let's use the official way!")
    print("=" * 70)
    
    # Step 1: Examine what's in the repository
    found_files = find_real_storydiffusion_integration()
    
    # Step 2: Try proper imports
    successful_imports = try_proper_storydiffusion_import()
    
    # Step 3: Create proper pipeline
    pipeline_result = create_proper_diffusers_pipeline()
    
    # Summary
    print(f"\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"📁 Found files: {len(found_files)}")
    print(f"✅ Successful imports: {len(successful_imports)}")
    
    if successful_imports:
        print(f"🔌 Working imports:")
        for import_stmt, desc in successful_imports:
            print(f"   - {desc}")
    
    if pipeline_result[0]:
        print(f"🚀 Pipeline created: {pipeline_result[1]}")
        print(f"💡 This is the REAL way to use StoryDiffusion with diffusers!")
    else:
        print(f"❌ No working pipeline found")
        print(f"💡 We may need to examine the code more carefully")
    
    # Next steps
    print(f"\n🎯 NEXT STEPS:")
    if pipeline_result[0]:
        print(f"✅ We have a working pipeline! Let's test generation.")
        print(f"🔧 Try generating with: pipeline_result[0](...)")
    else:
        print(f"🔍 We need to examine the actual StoryDiffusion code structure")
        print(f"📖 Check storydiffusionpipeline.py and understand the real API")
    
    return pipeline_result

if __name__ == "__main__":
    # Make sure we're in the right directory
    if not os.path.exists("utils"):
        print("❌ Not in StoryDiffusion directory. Please cd to StoryDiffusion-main first.")
        print("📁 Current directory:", os.getcwd())
    else:
        main()