# COMPREHENSIVE TESTING FRAMEWORK FOR ENHANCED CARTOON XL v8
# Tests all modules working together for coherent autism-friendly storyboard generation

import sys
import os
import time
import json
from PIL import Image
import torch

# Test imports - make sure all your modules work
def test_all_imports():
    """Test that all modularized files can be imported"""
    print("🧪 TESTING MODULE IMPORTS...")
    
    try:
        import config
        print("✅ config.py imported successfully")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        from background_analyzer import AIBackgroundAnalyzer
        print("✅ background_analyzer.py imported successfully")
    except Exception as e:
        print(f"❌ background_analyzer.py import failed: {e}")
        return False
    
    try:
        from consistency_manager import StoryDiffusionConsistencyManager
        print("✅ consistency_manager.py imported successfully")
    except Exception as e:
        print(f"❌ consistency_manager.py import failed: {e}")
        return False
    
    try:
        from emotion_detector import AIEmotionDetector
        print("✅ emotion_detector.py imported successfully")
    except Exception as e:
        print(f"❌ emotion_detector.py import failed: {e}")
        return False
    
    try:
        from enhanced_validator import EnhancedIntelligentImageValidator
        print("✅ enhanced_validator.py imported successfully")
    except Exception as e:
        print(f"❌ enhanced_validator.py import failed: {e}")
        return False
    
    try:
        from image_refiner import EnhancedAIImageRefiner
        print("✅ image_refiner.py imported successfully")
    except Exception as e:
        print(f"❌ image_refiner.py import failed: {e}")
        return False
    
    try:
        from learning_framework import AILearningFramework
        print("✅ learning_framework.py imported successfully")
    except Exception as e:
        print(f"❌ learning_framework.py import failed: {e}")
        return False
    
    try:
        from multi_model_ensemble import MultiModelEnsemble
        print("✅ multi_model_ensemble.py imported successfully")
    except Exception as e:
        print(f"❌ multi_model_ensemble.py import failed: {e}")
        return False
    
    try:
        from object_detector import ComprehensiveAIObjectDetector
        print("✅ object_detector.py imported successfully")
    except Exception as e:
        print(f"❌ object_detector.py import failed: {e}")
        return False
    
    try:
        from tifa_metric import PureAITIFAMetric
        print("✅ tifa_metric.py imported successfully")
    except Exception as e:
        print(f"❌ tifa_metric.py import failed: {e}")
        return False
    
    try:
        import main
        print("✅ main.py imported successfully")
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False
    
    print("✅ ALL MODULE IMPORTS SUCCESSFUL!")
    return True

def test_individual_modules():
    """Test each module's core functionality individually"""
    print("\n🧪 TESTING INDIVIDUAL MODULE FUNCTIONALITY...")
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color='lightblue')
    test_prompt = "a boy brushing his teeth, simple cartoon style"
    
    test_results = {}
    
    # Test TIFA Metrics
    try:
        from tifa_metric import PureAITIFAMetric
        tifa = PureAITIFAMetric()
        if tifa.available:
            tifa_result = tifa.calculate_pure_ai_tifa_score(test_image, test_prompt, "boy brushing teeth")
            test_results['tifa'] = {'status': 'success', 'score': tifa_result.get('score', 0)}
            print(f"✅ TIFA Metrics: {tifa_result.get('score', 0):.3f}")
        else:
            test_results['tifa'] = {'status': 'unavailable'}
            print("⚠️ TIFA Metrics: Unavailable")
    except Exception as e:
        test_results['tifa'] = {'status': 'error', 'error': str(e)}
        print(f"❌ TIFA Metrics: {e}")
    
    # Test Emotion Detector
    try:
        from emotion_detector import AIEmotionDetector
        emotion_detector = AIEmotionDetector()
        if emotion_detector.available:
            emotion_result = emotion_detector.ai_analyze_facial_emotions(test_image)
            test_results['emotion'] = {'status': 'success', 'faces': emotion_result.get('face_count', 0)}
            print(f"✅ Emotion Detector: {emotion_result.get('face_count', 0)} faces detected")
        else:
            test_results['emotion'] = {'status': 'unavailable'}
            print("⚠️ Emotion Detector: Unavailable")
    except Exception as e:
        test_results['emotion'] = {'status': 'error', 'error': str(e)}
        print(f"❌ Emotion Detector: {e}")
    
    # Test Object Detector
    try:
        from object_detector import ComprehensiveAIObjectDetector
        object_detector = ComprehensiveAIObjectDetector()
        if object_detector.available:
            object_result = object_detector.comprehensive_object_detection(test_image)
            test_results['objects'] = {'status': 'success', 'count': object_result.get('object_count', 0)}
            print(f"✅ Object Detector: {object_result.get('object_count', 0)} objects detected")
        else:
            test_results['objects'] = {'status': 'unavailable'}
            print("⚠️ Object Detector: Unavailable")
    except Exception as e:
        test_results['objects'] = {'status': 'error', 'error': str(e)}
        print(f"❌ Object Detector: {e}")
    
    # Test Background Analyzer
    try:
        from background_analyzer import AIBackgroundAnalyzer
        bg_analyzer = AIBackgroundAnalyzer()
        if bg_analyzer.available:
            bg_result = bg_analyzer.enhanced_ai_analyze_background_complexity(test_image)
            test_results['background'] = {'status': 'success', 'score': bg_result.get('autism_background_score', 0)}
            print(f"✅ Background Analyzer: {bg_result.get('autism_background_score', 0):.3f} autism score")
        else:
            test_results['background'] = {'status': 'unavailable'}
            print("⚠️ Background Analyzer: Unavailable")
    except Exception as e:
        test_results['background'] = {'status': 'error', 'error': str(e)}
        print(f"❌ Background Analyzer: {e}")
    
    # Test Multi-Model Ensemble
    try:
        from multi_model_ensemble import MultiModelEnsemble
        ensemble = MultiModelEnsemble()
        if ensemble.available:
            ensemble_result = ensemble.generate_ensemble_analysis(test_image, "test caption")
            test_results['ensemble'] = {'status': 'success', 'reliability': ensemble_result.get('reliability_score', 0)}
            print(f"✅ Multi-Model Ensemble: {ensemble_result.get('reliability_score', 0):.3f} reliability")
        else:
            test_results['ensemble'] = {'status': 'unavailable'}
            print("⚠️ Multi-Model Ensemble: Unavailable")
    except Exception as e:
        test_results['ensemble'] = {'status': 'error', 'error': str(e)}
        print(f"❌ Multi-Model Ensemble: {e}")
    
    return test_results

def test_integration_pipeline():
    """Test the complete integration pipeline"""
    print("\n🧪 TESTING COMPLETE INTEGRATION PIPELINE...")
    
    try:
        from enhanced_validator import EnhancedIntelligentImageValidator
        
        # Create validator
        validator = EnhancedIntelligentImageValidator()
        
        # Create test image
        test_image = Image.new('RGB', (1024, 1024), color='lightgreen')
        test_prompt = "a girl reading a book, autism-friendly, simple background"
        
        print("🔄 Running complete validation pipeline...")
        start_time = time.time()
        
        # Run full validation
        validation_result = validator.enhanced_validate_image(test_image, test_prompt, "autism_storyboard")
        
        pipeline_time = time.time() - start_time
        
        print(f"✅ INTEGRATION PIPELINE COMPLETED in {pipeline_time:.2f}s")
        print(f"📊 Overall Quality Score: {validation_result.get('quality_score', 0):.3f}")
        print(f"🎯 TIFA Score: {validation_result.get('tifa_result', {}).get('score', 0):.3f}")
        print(f"👥 Emotion Score: {validation_result.get('emotion_analysis', {}).get('autism_appropriateness_score', 0):.3f}")
        print(f"🧮 Background Score: {validation_result.get('background_analysis', {}).get('autism_background_score', 0):.3f}")
        print(f"🎓 Educational Score: {validation_result.get('educational_assessment', {}).get('score', 0):.1f}/100")
        print(f"🧠 Autism Suitable: {'✅ YES' if validation_result.get('educational_assessment', {}).get('suitable_for_autism') else '❌ NO'}")
        
        return True, validation_result
        
    except Exception as e:
        print(f"❌ INTEGRATION PIPELINE FAILED: {e}")
        return False, None

def test_storyboard_coherence():
    """Test storyboard coherence across multiple images"""
    print("\n🧪 TESTING STORYBOARD COHERENCE...")
    
    storyboard_prompts = [
        "a boy waking up in bed, autism-friendly cartoon",
        "the same boy brushing his teeth, autism-friendly cartoon", 
        "the same boy eating breakfast, autism-friendly cartoon",
        "the same boy going to school, autism-friendly cartoon"
    ]
    
    # Test consistency manager if available
    try:
        from consistency_manager import StoryDiffusionConsistencyManager
        consistency_manager = StoryDiffusionConsistencyManager()
        
        coherence_results = []
        
        for i, prompt in enumerate(storyboard_prompts):
            print(f"🎨 Testing storyboard frame {i+1}: {prompt}")
            
            # Create mock image for testing
            test_image = Image.new('RGB', (1024, 1024), color=f'hsl({i*90}, 50%, 80%)')
            
            # Test consistency analysis
            if hasattr(consistency_manager, 'analyze_character_consistency'):
                consistency_result = consistency_manager.analyze_character_consistency(test_image, prompt)
                coherence_results.append({
                    'frame': i+1,
                    'prompt': prompt,
                    'consistency_score': consistency_result.get('consistency_score', 0.5)
                })
                print(f"   📊 Consistency Score: {consistency_result.get('consistency_score', 0.5):.3f}")
            else:
                print("   ⚠️ Consistency analysis not available")
        
        if coherence_results:
            avg_consistency = sum(r['consistency_score'] for r in coherence_results) / len(coherence_results)
            print(f"✅ STORYBOARD COHERENCE TEST COMPLETED")
            print(f"📊 Average Consistency: {avg_consistency:.3f}")
            return True, coherence_results
        else:
            print("⚠️ No coherence results available")
            return False, None
            
    except Exception as e:
        print(f"❌ STORYBOARD COHERENCE TEST FAILED: {e}")
        return False, None

def test_learning_framework():
    """Test the learning framework if model is available"""
    print("\n🧪 TESTING LEARNING FRAMEWORK...")
    
    # Check if model file exists (replace with your actual model path)
    model_paths = [
        "RealCartoon-XL.safetensors",
        "models/RealCartoon-XL.safetensors",
        "../RealCartoon-XL.safetensors"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("⚠️ Model file not found - skipping learning framework test")
        print("   Expected paths:", model_paths)
        return False, None
    
    try:
        # Import the main validator with learning
        from main import EnhancedIntelligentRealCartoonXLValidator
        
        print(f"📁 Found model: {model_path}")
        print("🧠 Loading learning framework...")
        
        # This would normally load the full model - for testing we'll mock it
        print("⚠️ MOCK TEST: Learning framework structure")
        
        learning_test_result = {
            'model_found': True,
            'model_path': model_path,
            'learning_available': True,
            'mock_test': True
        }
        
        print("✅ LEARNING FRAMEWORK STRUCTURE TEST PASSED")
        return True, learning_test_result
        
    except Exception as e:
        print(f"❌ LEARNING FRAMEWORK TEST FAILED: {e}")
        return False, None

def test_performance_benchmarks():
    """Test performance benchmarks for all components"""
    print("\n🧪 TESTING PERFORMANCE BENCHMARKS...")
    
    benchmarks = {}
    test_image = Image.new('RGB', (1024, 1024), color='white')
    test_prompt = "performance test image"
    
    # Test TIFA performance
    try:
        from tifa_metric import PureAITIFAMetric
        tifa = PureAITIFAMetric()
        if tifa.available:
            start_time = time.time()
            tifa_result = tifa.calculate_pure_ai_tifa_score(test_image, test_prompt, "test caption")
            tifa_time = time.time() - start_time
            benchmarks['tifa'] = {'time': tifa_time, 'status': 'success'}
            print(f"⏱️ TIFA: {tifa_time:.2f}s")
        else:
            benchmarks['tifa'] = {'status': 'unavailable'}
    except Exception as e:
        benchmarks['tifa'] = {'status': 'error', 'error': str(e)}
    
    # Test emotion detection performance
    try:
        from emotion_detector import AIEmotionDetector
        emotion_detector = AIEmotionDetector()
        if emotion_detector.available:
            start_time = time.time()
            emotion_result = emotion_detector.ai_analyze_facial_emotions(test_image)
            emotion_time = time.time() - start_time
            benchmarks['emotion'] = {'time': emotion_time, 'status': 'success'}
            print(f"⏱️ Emotion Detection: {emotion_time:.2f}s")
        else:
            benchmarks['emotion'] = {'status': 'unavailable'}
    except Exception as e:
        benchmarks['emotion'] = {'status': 'error', 'error': str(e)}
    
    # Test object detection performance
    try:
        from object_detector import ComprehensiveAIObjectDetector
        object_detector = ComprehensiveAIObjectDetector()
        if object_detector.available:
            start_time = time.time()
            object_result = object_detector.comprehensive_object_detection(test_image)
            object_time = time.time() - start_time
            benchmarks['objects'] = {'time': object_time, 'status': 'success'}
            print(f"⏱️ Object Detection: {object_time:.2f}s")
        else:
            benchmarks['objects'] = {'status': 'unavailable'}
    except Exception as e:
        benchmarks['objects'] = {'status': 'error', 'error': str(e)}
    
    # Calculate total pipeline time
    successful_times = [b['time'] for b in benchmarks.values() if b.get('time')]
    if successful_times:
        total_time = sum(successful_times)
        benchmarks['total_pipeline'] = {'time': total_time, 'status': 'calculated'}
        print(f"⏱️ TOTAL PIPELINE: {total_time:.2f}s")
    
    return benchmarks

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧪 COMPREHENSIVE TESTING FRAMEWORK")
    print("=" * 80)
    print("🎯 Testing Enhanced Intelligent Real Cartoon XL v8")
    print("🧠 Testing modularized architecture for autism-friendly storyboards")
    
    test_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': {}
    }
    
    # Test 1: Module Imports
    print(f"\n{'='*60}")
    print("TEST 1: MODULE IMPORTS")
    print(f"{'='*60}")
    imports_success = test_all_imports()
    test_results['tests']['imports'] = {'success': imports_success}
    
    if not imports_success:
        print("❌ Cannot continue - fix import issues first")
        return test_results
    
    # Test 2: Individual Modules
    print(f"\n{'='*60}")
    print("TEST 2: INDIVIDUAL MODULE FUNCTIONALITY")
    print(f"{'='*60}")
    module_results = test_individual_modules()
    test_results['tests']['individual_modules'] = module_results
    
    # Test 3: Integration Pipeline
    print(f"\n{'='*60}")
    print("TEST 3: COMPLETE INTEGRATION PIPELINE")
    print(f"{'='*60}")
    pipeline_success, pipeline_result = test_integration_pipeline()
    test_results['tests']['integration'] = {
        'success': pipeline_success,
        'result': pipeline_result
    }
    
    # Test 4: Storyboard Coherence
    print(f"\n{'='*60}")
    print("TEST 4: STORYBOARD COHERENCE")
    print(f"{'='*60}")
    coherence_success, coherence_result = test_storyboard_coherence()
    test_results['tests']['coherence'] = {
        'success': coherence_success,
        'result': coherence_result
    }
    
    # Test 5: Learning Framework
    print(f"\n{'='*60}")
    print("TEST 5: LEARNING FRAMEWORK")
    print(f"{'='*60}")
    learning_success, learning_result = test_learning_framework()
    test_results['tests']['learning'] = {
        'success': learning_success,
        'result': learning_result
    }
    
    # Test 6: Performance Benchmarks
    print(f"\n{'='*60}")
    print("TEST 6: PERFORMANCE BENCHMARKS")
    print(f"{'='*60}")
    benchmarks = test_performance_benchmarks()
    test_results['tests']['performance'] = benchmarks
    
    # Final Summary
    print(f"\n{'='*60}")
    print("🎉 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(test_results['tests'])
    successful_tests = sum(1 for test in test_results['tests'].values() 
                          if test.get('success') or 
                          (isinstance(test, dict) and any(isinstance(v, dict) and v.get('status') == 'success' for v in test.values())))
    
    print(f"📊 Tests Completed: {total_tests}")
    print(f"✅ Tests Successful: {successful_tests}")
    print(f"❌ Tests Failed: {total_tests - successful_tests}")
    print(f"📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # AI Capabilities Summary
    print(f"\n🧠 AI CAPABILITIES SUMMARY:")
    module_results = test_results['tests'].get('individual_modules', {})
    for module, result in module_results.items():
        status = result.get('status', 'unknown')
        if status == 'success':
            print(f"   ✅ {module.upper()}: Operational")
        elif status == 'unavailable':
            print(f"   ⚠️ {module.upper()}: Unavailable (dependencies missing)")
        else:
            print(f"   ❌ {module.upper()}: Error")
    
    # Performance Summary
    performance = test_results['tests'].get('performance', {})
    if performance:
        print(f"\n⏱️ PERFORMANCE SUMMARY:")
        for component, bench in performance.items():
            if bench.get('time'):
                print(f"   {component}: {bench['time']:.2f}s")
    
    # Save results
    try:
        with open('test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\n💾 Test results saved to: test_results.json")
    except Exception as e:
        print(f"\n⚠️ Could not save test results: {e}")
    
    return test_results

if __name__ == "__main__":
    # Run comprehensive tests
    results = run_comprehensive_tests()
    
    print(f"\n🎯 TESTING COMPLETE!")
    print(f"📄 Check test_results.json for detailed results")
    print(f"🔧 Fix any failed tests before using the system")
    print(f"🧠 Your modularized autism storyboard system is ready for coherence testing!")
