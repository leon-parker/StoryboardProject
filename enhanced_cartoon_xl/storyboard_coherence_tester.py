#!/usr/bin/env python3
"""
Simple Autism Storyboard Pipeline Tester
Just imports your pipeline and tests it with autism-friendly prompts
"""

import time
from main import generate_image_with_validation

def test_autism_storyboard_pipeline():
    """Test your autism storyboard pipeline with sample prompts"""
    
    print("🎬 AUTISM STORYBOARD PIPELINE TESTER")
    print("📖 Testing consistent character story: Boy's Morning Routine")
    print("=" * 60)
    
    # Consistent character autism storyboard - Same boy throughout
    test_prompts = [
        "a young boy with short brown hair waking up in bed, stretching arms, peaceful expression, simple cartoon style, autism-friendly, clean bedroom",
        "the same boy with short brown hair brushing teeth at bathroom sink, focused expression, simple cartoon style, autism-friendly, clean bathroom",
        "the same boy with short brown hair eating breakfast at kitchen table, happy expression, simple cartoon style, autism-friendly, minimal kitchen",
        "the same boy with short brown hair washing hands at sink, concentrated expression, simple cartoon style, autism-friendly, simple background"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎨 SCENE {i}/4: {prompt}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Use your actual pipeline
        result = generate_image_with_validation(
            prompt=prompt,
            target_score=0.75,
            attempts=2
        )
        
        generation_time = time.time() - start_time
        
        if result['success']:
            print(f"✅ SUCCESS")
            print(f"📊 Score: {result['score']:.3f}")
            print(f"🎯 Autism Suitable: {result['autism_suitable']}")
            print(f"⚡ Time: {generation_time:.1f}s")
            
            # Save image with descriptive scene name
            scene_names = ["waking_up", "brushing_teeth", "eating_breakfast", "washing_hands"]
            filename = f"boy_morning_routine_{scene_names[i-1]}.png"
            result['image'].save(filename)
            print(f"💾 Saved: {filename}")
            
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        
        results.append({
            'prompt': prompt,
            'success': result['success'],
            'score': result.get('score', 0),
            'time': generation_time
        })
    
    # Summary
    print(f"\n🎉 STORYBOARD TEST SUMMARY")
    print("📖 Boy's Morning Routine - Character Consistency Test")
    print("=" * 60)
    successful = sum(1 for r in results if r['success'])
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_score = sum(r['score'] for r in results if r['success']) / max(successful, 1)
    
    print(f"✅ Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    print(f"📊 Average Score: {avg_score:.3f}")
    print(f"⚡ Average Time: {avg_time:.1f}s per image")
    print(f"💾 Storyboard scenes saved as: boy_morning_routine_*.png")
    
    return results

if __name__ == "__main__":
    test_autism_storyboard_pipeline()