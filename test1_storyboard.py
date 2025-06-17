"""
Test Storyboard - IP-Adapter Character Consistency Verification
Simple test to verify IP-Adapter character consistency throughout a story
"""

import os
import time
from cartoon_Pipeline_IP_Adapter import CartoonPipelineWithIPAdapter


def test_character_consistency():
    """Test character consistency with a simple morning routine storyboard"""
    
    print("🧪 IP-ADAPTER CHARACTER CONSISTENCY TEST")
    print("=" * 50)
    
    # ===== CORRECTED PATHS FOR IP-ADAPTER =====
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    main_project_dir = os.path.dirname(parent_dir)
    
    # FIXED: Model is in the models subfolder
    MODEL_PATH = os.path.join(main_project_dir, "models", "realcartoonxl_v7.safetensors")
    IP_ADAPTER_PATH = os.path.join(main_project_dir, "ip-adapter_sdxl.bin")  # ENABLE IP-ADAPTER!
    
    print(f"🔍 Looking for model at: {MODEL_PATH}")
    print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"🎭 Looking for IP-Adapter at: {IP_ADAPTER_PATH}")
    print(f"📁 IP-Adapter exists: {os.path.exists(IP_ADAPTER_PATH)}")
    
    # If model not found, try common locations
    if not os.path.exists(MODEL_PATH):
        print("🔍 Searching for model in common locations...")
        
        # Try current directory
        current_model = os.path.join(os.getcwd(), "realcartoonxl_v7.safetensors")
        if os.path.exists(current_model):
            MODEL_PATH = current_model
            print(f"✅ Found model in current directory: {MODEL_PATH}")
        else:
            # Search for any .safetensors file
            for root, dirs, files in os.walk(main_project_dir):
                for file in files:
                    if file.endswith('.safetensors') and 'realcartoon' in file.lower():
                        MODEL_PATH = os.path.join(root, file)
                        print(f"✅ Found model: {MODEL_PATH}")
                        break
                if MODEL_PATH and os.path.exists(MODEL_PATH):
                    break
    
    # If IP-Adapter not found, search for it
    if not os.path.exists(IP_ADAPTER_PATH):
        print("🔍 Searching for IP-Adapter in common locations...")
        for root, dirs, files in os.walk(main_project_dir):
            for file in files:
                if 'ip-adapter' in file.lower() and file.endswith('.bin'):
                    IP_ADAPTER_PATH = os.path.join(root, file)
                    print(f"✅ Found IP-Adapter: {IP_ADAPTER_PATH}")
                    break
            if IP_ADAPTER_PATH and os.path.exists(IP_ADAPTER_PATH):
                break
    
    # ===============================
    
    # Test storyboard prompts - morning routine for autism education
    test_prompts = [
        "young boy waking up in bed, stretching arms, cartoon style, simple clean art",
        "same young boy sitting on edge of bed, cartoon style, simple clean art", 
        "same young boy brushing teeth in bathroom, cartoon style, simple clean art",
        "same young boy getting dressed in bedroom, cartoon style, simple clean art",
        "same young boy eating breakfast at kitchen table, cartoon style, simple clean art"
    ]
    
    print(f"📖 Testing with {len(test_prompts)} story prompts")
    print(f"🎯 Goal: SAME CHARACTER throughout all scenes using IP-Adapter")
    print()
    
    # Check if required files exist
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found!")
        print(f"   Searched path: {MODEL_PATH}")
        print("\n📋 Please ensure your model file exists")
        return
    
    if not os.path.exists(IP_ADAPTER_PATH):
        print("❌ IP-Adapter file not found!")
        print(f"   Searched path: {IP_ADAPTER_PATH}")
        print("\n📋 Please ensure ip-adapter_sdxl.bin is in your project directory")
        print("   Download from: https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin")
        return
    
    # Initialize pipeline WITH IP-ADAPTER
    print("🔧 Initializing pipeline with IP-Adapter...")
    pipeline = CartoonPipelineWithIPAdapter(
        model_path=MODEL_PATH,
        ip_adapter_path=IP_ADAPTER_PATH  # ENABLE IP-ADAPTER
    )
    
    if not pipeline.available:
        print("❌ Pipeline not available!")
        print("📋 Requirements:")
        print("   1. Model file exists ✅" if os.path.exists(MODEL_PATH) else "   1. Model file missing ❌")
        print("   2. IP-Adapter file exists ✅" if os.path.exists(IP_ADAPTER_PATH) else "   2. IP-Adapter file missing ❌")
        print("   3. Install: pip install diffusers[ip-adapter] transformers torch")
        return
    
    # Configure for MAXIMUM CHARACTER CONSISTENCY with IP-Adapter
    pipeline.update_config(
        selection={
            "use_consistency": True,
            "use_ip_adapter": True,      # ENABLED!
            "quality_weight": 0.2,       # Lower quality priority
            "consistency_weight": 0.3,   # Medium CLIP consistency
            "ip_adapter_weight": 0.5     # HIGHEST IP-Adapter priority
        },
        ip_adapter={
            "character_weight": 0.50,    # MAXIMUM character consistency
            "update_reference_from_best": True,
            "fallback_to_clip": True
        }
    )
    
    print("✅ Pipeline configured for MAXIMUM IP-Adapter character consistency")
    print()
    
    # Create output directory
    output_dir = "test_storyboard_output2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate storyboard
    start_time = time.time()
    
    print("🎨 Generating IP-Adapter character-consistent storyboard...")
    sequence_result = pipeline.generate_storyboard_sequence(
        prompts_list=test_prompts,
        character_reference_image=None,  # Let first image establish reference
        iterations_per_prompt=2,         # Single iteration for speed
        images_per_iteration=3           # Generate 3 options per prompt
    )
    
    total_time = time.time() - start_time
    
    if not sequence_result:
        print("❌ Storyboard generation failed!")
        return
    
    # Process results
    print(f"\n🎉 IP-ADAPTER STORYBOARD COMPLETE! Generated in {total_time:.1f}s")
    print("=" * 50)
    
    sequence_data = sequence_result["sequence_results"]
    consistency_report = sequence_result["consistency_report"]
    ip_status = sequence_result["ip_adapter_status"]
    
    # Save images and show results
    print("💾 Saving storyboard frames...")
    for i, frame_data in enumerate(sequence_data):
        # Save image
        filename = f"{output_dir}/frame_{i+1:02d}.png"
        frame_data["best_image"].save(filename)
        
        # Show frame info
        prompt_preview = frame_data["prompt"][:60] + "..." if len(frame_data["prompt"]) > 60 else frame_data["prompt"]
        ip_used = "✅ IP-ADAPTER" if frame_data.get("used_ip_adapter", False) else "❌ CLIP-ONLY"
        
        print(f"   Frame {i+1}: {prompt_preview}")
        print(f"           Score: {frame_data['final_score']:.3f} | Method: {ip_used}")
    
    print()
    
    # Show IP-Adapter specific analysis
    print("🎭 IP-ADAPTER ANALYSIS:")
    print(f"   Total frames generated: {len(sequence_data)}")
    print(f"   Character reference established: {sequence_result['character_reference_established']}")
    print(f"   IP-Adapter used: {sum(1 for r in sequence_data if r.get('used_ip_adapter', False))}/{len(sequence_data)} times")
    print(f"   IP-Adapter available: {ip_status['available']}")
    print(f"   Character reference set: {ip_status['character_reference_set']}")
    
    if consistency_report["available"]:
        print(f"\n📊 CLIP Consistency (backup metric):")
        print(f"   Grade: {consistency_report['consistency_grade']}")
        print(f"   Average score: {consistency_report['average_consistency']:.3f}")
        print(f"   Range: {consistency_report['min_consistency']:.3f} - {consistency_report['max_consistency']:.3f}")
    
    print(f"\n📁 All frames saved to: {output_dir}/")
    
    # Generate test report
    generate_test_report(sequence_result, output_dir, total_time)
    
    # Show IP-Adapter specific results
    ip_usage_count = sum(1 for r in sequence_data if r.get('used_ip_adapter', False))
    total_frames = len(sequence_data)
    
    print("\n🎯 IP-ADAPTER TEST RESULTS:")
    if ip_usage_count >= total_frames - 1:  # All except possibly first frame
        print("   ✅ SUCCESS: IP-Adapter used for character consistency")
        print("   🎭 Expected: Same character appearance across all frames")
    else:
        print(f"   ⚠️  WARNING: IP-Adapter only used {ip_usage_count}/{total_frames} times")
        print("   🔧 Check IP-Adapter installation and file paths")
    
    print("\n🔍 NEXT STEPS:")
    print("1. ✅ Check generated images for IDENTICAL character features")
    print("2. 🎭 Look for consistent: hair, face, clothing, body type")
    print("3. 📊 Compare with CLIP-only results for improvement")
    print("4. 🎯 Character should be IDENTICAL across all 5 frames!")


def generate_test_report(sequence_result, output_dir, generation_time):
    """Generate a detailed IP-Adapter test report"""
    
    report_path = os.path.join(output_dir, "ip_adapter_character_consistency_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("IP-ADAPTER CHARACTER CONSISTENCY TEST REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generation Time: {generation_time:.1f} seconds\n")
        f.write(f"Total Frames: {len(sequence_result['sequence_results'])}\n")
        f.write(f"Character Reference Established: {sequence_result['character_reference_established']}\n\n")
        
        # IP-Adapter status
        ip_status = sequence_result.get("ip_adapter_status", {"available": False})
        f.write("IP-ADAPTER STATUS:\n")
        f.write(f"  Available: {ip_status.get('available', False)}\n")
        f.write(f"  Character Reference Set: {ip_status.get('character_reference_set', False)}\n")
        f.write(f"  Configuration: {ip_status.get('config', {})}\n\n")
        
        # Consistency analysis
        consistency = sequence_result["consistency_report"]
        if consistency["available"]:
            f.write("CLIP CONSISTENCY ANALYSIS (Backup Metric):\n")
            f.write(f"  Grade: {consistency['consistency_grade']}\n")
            f.write(f"  Average Score: {consistency['average_consistency']:.3f}\n")
            f.write(f"  Score Range: {consistency['min_consistency']:.3f} - {consistency['max_consistency']:.3f}\n")
            f.write(f"  Standard Deviation: {consistency['consistency_std']:.3f}\n\n")
        
        # Frame details
        f.write("FRAME DETAILS:\n")
        for i, frame in enumerate(sequence_result['sequence_results']):
            f.write(f"\nFrame {i+1}:\n")
            f.write(f"  Prompt: {frame['prompt']}\n")
            f.write(f"  Score: {frame['final_score']:.3f}\n")
            f.write(f"  IP-Adapter Used: {frame.get('used_ip_adapter', False)}\n")
            f.write(f"  File: frame_{i+1:02d}.png\n")
        
        # IP-Adapter specific evaluation
        f.write("\nIP-ADAPTER TEST EVALUATION:\n")
        ip_usage = sum(1 for r in sequence_result['sequence_results'] if r.get('used_ip_adapter', False))
        total_frames = len(sequence_result['sequence_results'])
        
        f.write(f"IP-Adapter Usage: {ip_usage}/{total_frames} frames\n")
        
        if ip_usage >= total_frames - 1:  # All frames except possibly first
            f.write("✅ PASS: IP-Adapter used for character consistency\n")
            f.write("🎭 EXPECTED RESULT: Identical character across all frames\n")
        else:
            f.write("⚠️  WARNING: IP-Adapter not used consistently\n")
            f.write("🔧 RECOMMENDATION: Check IP-Adapter installation\n")
        
        if consistency["available"] and consistency['average_consistency'] > 0.8:
            f.write("✅ EXCELLENT: Very high visual consistency\n")
        elif consistency["available"] and consistency['average_consistency'] > 0.7:
            f.write("✅ GOOD: High visual consistency\n")
        elif consistency["available"] and consistency['average_consistency'] > 0.6:
            f.write("⚠️  MODERATE: Moderate visual consistency\n")
        elif consistency["available"]:
            f.write("❌ POOR: Low visual consistency\n")
        else:
            f.write("❌ ERROR: Unable to measure consistency\n")
    
    print(f"📊 IP-Adapter test report saved: {report_path}")


def quick_single_test():
    """Quick test with IP-Adapter for single image"""
    
    # Auto-find paths
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    main_project_dir = os.path.dirname(parent_dir)
    # FIXED: Model is in the models subfolder
    MODEL_PATH = os.path.join(main_project_dir, "models", "realcartoonxl_v7.safetensors")
    IP_ADAPTER_PATH = os.path.join(main_project_dir, "ip-adapter_sdxl.bin")
    
    print("🧪 QUICK IP-ADAPTER TEST")
    print("=" * 30)
    print(f"🔍 Model path: {MODEL_PATH}")
    print(f"📁 Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"🎭 IP-Adapter path: {IP_ADAPTER_PATH}")
    print(f"📁 IP-Adapter exists: {os.path.exists(IP_ADAPTER_PATH)}")
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found!")
        return
    
    if not os.path.exists(IP_ADAPTER_PATH):
        print("❌ IP-Adapter not found!")
        return
    
    pipeline = CartoonPipelineWithIPAdapter(MODEL_PATH, IP_ADAPTER_PATH)
    
    if not pipeline.available:
        print("❌ Pipeline not available!")
        return
    
    result = pipeline.generate_single_image(
        prompt="young boy smiling, cartoon style, simple clean art",
        num_images=2
    )
    
    if result:
        result["image"].save("quick_ip_adapter_test.png")
        print(f"✅ Test image saved: quick_ip_adapter_test.png")
        print(f"   Score: {result['score']:.3f}")
        print(f"   IP-Adapter used: {result.get('used_ip_adapter', False)}")
    else:
        print("❌ Test failed")


if __name__ == "__main__":
    print("🎨 IP-Adapter Character Consistency Test Suite")
    print("For autism education storyboards with IDENTICAL characters")
    print()
    
    choice = input("Choose test:\n1. Full IP-Adapter storyboard test\n2. Quick IP-Adapter single image test\nChoice (1-2): ")
    
    if choice == "1":
        test_character_consistency()
    elif choice == "2":
        quick_single_test()
    else:
        print("Invalid choice - running full IP-Adapter test...")
        test_character_consistency()