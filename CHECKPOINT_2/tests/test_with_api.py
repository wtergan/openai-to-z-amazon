"""
Simple test script for testing with real OpenRouter API.
Run this after setting up your API keys in environment variables.
"""
import os
from test_anomaly_pipeline import run_full_pipeline_test
from dataset_fetching import make_bbox

def test_with_real_api():
    """Test the pipeline with real OpenRouter API calls."""
    
    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your_api_key_here'")
        return False
    
    print("Testing anomaly detection pipeline with real OpenRouter API...")
    print("Using a small bounding box and only 2 cells to minimize costs.")
    
    # Use a small bounding box for testing
    bbox = make_bbox(-60.2, -3.2, -60.0, -3.0)
    
    # Run the full pipeline with LLM assessment
    try:
        results = run_full_pipeline_test(
            bbox=bbox,
            year=2022,
            n_top=2,  # Only test with 2 cells to minimize API costs
            provider="openrouter",
            model_name="google/gemma-3-27b-it",  # A relatively affordable model
            test_llm=True,
            test_batch=True
        )
        
        if 'error' in results:
            print(f"Error in pipeline: {results['error']}")
            return False
        
        # Display results
        print("\n" + "="*60)
        print("API TEST RESULTS")
        print("="*60)
        
        if 'llm_assessment' in results:
            for method, assessment in results['llm_assessment'].items():
                print(f"\n{method} results:")
                if 'individual' in method and isinstance(assessment, list):
                    for i, result in enumerate(assessment, 1):
                        print(f"  Cell {i} ({result['cell_id']}):")
                        response = result['llm_response']
                        print(f"    Response: {response[:200]}...")
                elif 'batch' in method and isinstance(assessment, dict):
                    print(f"  Batch assessment for {assessment['num_cells']} cells:")
                    response = assessment['llm_response']
                    print(f"    Response: {response[:200]}...")
        
        print(f"\nCheck the log file 'llm_prompt_log.jsonl' for complete LLM interactions.")
        return True
        
    except Exception as e:
        print(f"Error during API test: {e}")
        return False

if __name__ == "__main__":
    success = test_with_real_api()
    if success:
        print("\n✅ API test completed successfully!")
    else:
        print("\n❌ API test failed.") 