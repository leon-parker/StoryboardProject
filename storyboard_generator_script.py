# COMPLETE 5-SCENE SCHOOL BOY STORYBOARD GENERATOR
import os
import time as time_module
from pathlib import Path
from datetime import datetime
from lightweight_RealCartoon import StreamlinedRealCartoonXL

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = "models/realcartoonxl_v7.safetensors"  # UPDATE THIS PATH
OUTPUT_DIR = "schoolboy_storyboard"

# =============================================================================
# 5 SCHOOL BOY DAILY ROUTINE SCENES
# =============================================================================

DAILY_ROUTINE_SCENES = [
    "school boy eating breakfast at kitchen table, cereal bowl, cartoon style, friendly expression",
    "school boy writing in notebook at classroom desk, focused learning, same young boy",
    "school boy eating lunch in cafeteria, tray with food, consistent character,", 
    "school boy doing homework at home desk, pencils and books, same character design",
    "school boy in pajamas reading bedtime story, cozy bedroom, peaceful evening"
]

SCENE_NAMES = ["breakfast", "classroom", "lunch", "homework", "bedtime"]
SCENE_TIMES = ["7:00 AM", "9:00 AM", "12:00 PM", "4:00 PM", "8:00 PM"]

# =============================================================================
# COMPLETE 5-SCENE STORYBOARD GENERATION
# =============================================================================

def generate_complete_storyboard():
    print("🎓 SCHOOL BOY DAILY ROUTINE - 5 SCENES")
    print("=" * 60)
    
    # Show the schedule
    print("📅 DAILY ROUTINE SCHEDULE:")
    for i, (time, name, scene) in enumerate(zip(SCENE_TIMES, SCENE_NAMES, DAILY_ROUTINE_SCENES)):
        print(f"   {i+1}. {time} - {name.title()}")
    print()
    
    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / f"daily_routine_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path}")
    
    # Load generator
    print("\n🚀 Loading Real Cartoon XL v7...")
    generator = StreamlinedRealCartoonXL(MODEL_PATH)
    
    if not generator.available:
        print("❌ Failed to load generator")
        return
    
    # Generate complete storyboard with consistency
    print(f"\n📚 Starting 5-scene storyboard generation...")
    print(f"🔄 2 iterations per scene, 3 images per iteration")
    start_time = time_module.time()
    
    storyboard_result = generator.generate_storyboard_sequence(
        DAILY_ROUTINE_SCENES,
        iterations_per_prompt=2,  # 2 iterations for speed
        images_per_iteration=3     # 3 images per iteration
    )
    
    total_time = time_module.time() - start_time
    
    # Save all 5 scene images
    print(f"\n💾 Saving 5 final storyboard images...")
    
    for i, scene_result in enumerate(storyboard_result["storyboard_results"]):
        best_result = scene_result["best_result"]
        
        # Create detailed metadata for each scene
        metadata = {
            "scene_number": i + 1,
            "scene_name": SCENE_NAMES[i],
            "scene_time": SCENE_TIMES[i],
            "prompt": best_result["prompt"],
            "final_prompt": scene_result["final_prompt"],
            "tifa_score": best_result["tifa_score"],
            "clip_similarity": best_result.get("clip_similarity", 0.0),
            "caption": best_result.get("caption", ""),
            "iterations_completed": len(scene_result["history"]),
            "score_improvement": scene_result.get("score_improvement", 0.0),
            "generation_timestamp": datetime.now().isoformat()
        }
        
        # Save the image
        filename = f"scene_{i+1:02d}_{SCENE_NAMES[i]}_{SCENE_TIMES[i].replace(':', '')}"
        image_path = output_path / f"{filename}.png"
        best_result["best_image"].save(image_path)
        
        # Save metadata
        metadata_path = output_path / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   💾 Scene {i+1} ({SCENE_NAMES[i]}): TIFA {best_result['tifa_score']:.3f} → {image_path.name}")
    
    # Save complete storyboard summary
    summary = {
        "storyboard_type": "School Boy Daily Routine",
        "generation_timestamp": datetime.now().isoformat(),
        "total_generation_time_minutes": total_time / 60,
        "scenes": len(DAILY_ROUTINE_SCENES),
        "results": {
            "average_tifa": storyboard_result["average_tifa"],
            "tifa_scores": storyboard_result["tifa_scores"],
            "consistency_report": storyboard_result["consistency_report"],
            "total_images_generated": storyboard_result["generation_summary"]["total_images_generated"],
            "selected_images": storyboard_result["generation_summary"]["selected_images"]
        },
        "daily_schedule": [
            {
                "scene": i+1,
                "time": SCENE_TIMES[i], 
                "activity": SCENE_NAMES[i],
                "prompt": scene
            } for i, scene in enumerate(DAILY_ROUTINE_SCENES)
        ]
    }
    
    summary_path = output_path / "storyboard_summary.json"
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2, default=str)
    
    # Print final results
    print(f"\n🎉 COMPLETE STORYBOARD GENERATED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🎯 Average TIFA score: {storyboard_result['average_tifa']:.3f}")
    print(f"📈 TIFA progression: {[f'{score:.3f}' for score in storyboard_result['tifa_scores']]}")
    
    # Show consistency results
    if storyboard_result["consistency_report"]["available"]:
        consistency = storyboard_result["consistency_report"]
        print(f"🧠 Character consistency: {consistency['consistency_grade']} ({consistency['average_consistency']:.3f})")
        print(f"   Consistency range: {consistency['min_consistency']:.3f} - {consistency['max_consistency']:.3f}")
    
    print(f"📁 All files saved to: {output_path}")
    print(f"🎨 Generated {storyboard_result['generation_summary']['total_images_generated']} total images")
    print(f"🏆 Selected best 5 images for storyboard")
    
    # Show final scene breakdown
    print(f"\n📖 FINAL STORYBOARD SCENES:")
    for i, (time, name) in enumerate(zip(SCENE_TIMES, SCENE_NAMES)):
        tifa_score = storyboard_result['tifa_scores'][i]
        print(f"   {i+1}. {time} - {name.title()}: TIFA {tifa_score:.3f}")
    
    print(f"\n✅ School boy daily routine storyboard complete!")

def generate_single_test():
    """Generate just one scene for testing"""
    print("🧪 TESTING SINGLE SCENE")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    generator = StreamlinedRealCartoonXL(MODEL_PATH)
    if not generator.available:
        print("❌ Failed to load generator")
        return
    
    # Test first scene
    test_prompt = DAILY_ROUTINE_SCENES[0]
    print(f"🎨 Testing: {test_prompt}")
    
    result = generator.iterative_improvement_generation(test_prompt, iterations=3, images_per_iteration=3)
    
    if result is not None:
        print(f"✅ SUCCESS! TIFA: {result['best_result']['tifa_score']:.3f}")
        
        # Save test image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(OUTPUT_DIR) / f"test_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        result["best_result"]["best_image"].save(output_path / "test_breakfast.png")
        print(f"💾 Saved: {output_path / 'test_breakfast.png'}")
    else:
        print("❌ FAILED - result is None")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎓 SCHOOL BOY STORYBOARD GENERATOR")
    print("Choose generation mode:")
    print("1. Complete 5-scene storyboard (recommended)")
    print("2. Single scene test")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        generate_single_test()
    else:
        generate_complete_storyboard()