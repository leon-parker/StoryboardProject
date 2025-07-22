# generate_storyboard.py - Your specific generation script
from GPT_RealCartoon import RealCartoonXLAutismValidator
import os

def main():
    # Initialize the validator with your model
    model_path = r"C:\RuinedFooocus-main\models\checkpoints\realcartoonxl_v7.safetensors"
    validator = RealCartoonXLAutismValidator(model_path)
    
    # Define your autism storyboard images
    storyboard_prompts = [
        {
            "name": "morning_routine_1",
            "prompt": "young boy with brown hair waking up in bed, cartoon style, simple bedroom background",
            "description": "Morning routine - waking up"
        },
        {
            "name": "morning_routine_2", 
            "prompt": "boy brushing teeth in bathroom mirror, cartoon style, simple background",
            "description": "Morning routine - brushing teeth"
        },
        {
            "name": "morning_routine_3",
            "prompt": "child eating breakfast cereal at kitchen table, cartoon style, simple background",
            "description": "Morning routine - eating breakfast"
        },
        {
            "name": "school_prep_1",
            "prompt": "boy putting on school backpack by front door, cartoon style, getting ready",
            "description": "School preparation - backpack"
        },
        {
            "name": "school_prep_2",
            "prompt": "child walking to school bus, cartoon style, simple street background",
            "description": "School preparation - going to bus"
        }
    ]
    
    # Output directory
    output_dir = "./autism_storyboard_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each storyboard image
    results = []
    for i, scene in enumerate(storyboard_prompts):
        print(f"\n{'='*60}")
        print(f"🎨 GENERATING SCENE {i+1}/5: {scene['description']}")
        print(f"{'='*60}")
        
        # Generate with AI refinement
        result = validator.generate_storyboard_image_with_refinement(
            prompt=scene['prompt'],
            attempts=3,
            target_score=0.65,
            enable_refinement=True
        )
        
        if result:
            # Save the image
            image_path = f"{output_dir}/{scene['name']}.png"
            result['image'].save(image_path)
            
            # Save validation report
            report_path = f"{output_dir}/{scene['name']}_report.json"
            validation_copy = result['validation'].copy()
            # Handle numpy arrays for JSON
            validation_copy['face_analysis'] = {
                'count': validation_copy['face_count'],
                'positions': [pos for pos in validation_copy['face_positions']]
            }
            
            import json
            with open(report_path, 'w') as f:
                json.dump(validation_copy, f, indent=2)
            
            # Print summary
            print(f"\n📊 SCENE {i+1} RESULTS:")
            print(f"💾 Saved: {image_path}")
            print(f"📊 Score: {result['validation']['quality_score']:.3f}")
            print(f"🔧 Refined: {'✅ YES' if result.get('refined') else '❌ NO'}")
            if result.get('refined'):
                print(f"📈 Improvement: +{result.get('improvement', 0):.3f}")
            print(f"🧠 Autism Suitable: {'✅ YES' if result['validation']['autism_suitability']['suitable'] else '❌ NO'}")
            print(f"👤 Characters: {result['validation']['face_count']}")
            print(f"🎯 Recommendations: {result['validation']['recommendations']}")
            
            results.append(result)
        else:
            print(f"❌ Failed to generate scene {i+1}")
    
    # Final summary
    print(f"\n🎉 STORYBOARD GENERATION COMPLETE!")
    print(f"📊 Successfully generated: {len(results)}/5 scenes")
    
    suitable_count = sum(1 for r in results if r['validation']['autism_suitability']['suitable'])
    print(f"🧠 Autism suitable: {suitable_count}/{len(results)} scenes")
    
    refined_count = sum(1 for r in results if r.get('refined'))
    print(f"🔧 AI refined: {refined_count}/{len(results)} scenes")
    
    avg_score = sum(r['validation']['quality_score'] for r in results) / len(results) if results else 0
    print(f"📈 Average quality score: {avg_score:.3f}")
    
    print(f"📁 Check {output_dir}/ for all images and reports")

if __name__ == "__main__":
    main()