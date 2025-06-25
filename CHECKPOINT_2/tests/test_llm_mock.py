"""
Mock LLM test to verify message structure and batch processing without API calls.
"""
import json
from typing import Dict, List, Any
import pandas as pd

from test_anomaly_pipeline import run_full_pipeline_test, create_mock_lidar_s2_context
from dataset_fetching import make_bbox
from model_integration import build_messages, build_batch_messages

def test_message_structure():
    """Test the message structure for both individual and batch approaches."""
    print("Testing LLM message structure...")
    
    # Get some test data
    bbox = make_bbox(-60.5, -3.5, -60.0, -3.0)
    results = run_full_pipeline_test(bbox, 2022, n_top=3, test_llm=False)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    # Get top cells from weighted scoring
    if 'weighted' in results['scoring']:
        top_cells = results['scoring']['weighted'].to_dict('records')[:2]  # Just use 2 cells for testing
    else:
        print("No weighted scoring results available")
        return
    
    # Create mock regional context
    lidar_s2_ctx = create_mock_lidar_s2_context()
    
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL MESSAGE STRUCTURE")
    print("=" * 60)
    
    for i, cell in enumerate(top_cells, 1):
        print(f"\n--- Cell {i} Message ---")
        messages = build_messages(cell, lidar_s2_ctx)
        
        print(f"Number of messages: {len(messages)}")
        for j, msg in enumerate(messages):
            print(f"Message {j+1} role: {msg['role']}")
            if msg['role'] == 'system':
                print(f"System content length: {len(msg['content'])} chars")
                print(f"System content preview: {msg['content'][:100]}...")
            elif msg['role'] == 'user':
                print(f"User content items: {len(msg['content'])}")
                for k, item in enumerate(msg['content']):
                    if item['type'] == 'text':
                        print(f"  Item {k+1}: text ({len(item['text'])} chars)")
                        print(f"    Preview: {item['text'][:100]}...")
                    elif item['type'] == 'image_url':
                        print(f"  Item {k+1}: image_url")
    
    print("\n" + "=" * 60)
    print("TESTING BATCH MESSAGE STRUCTURE")
    print("=" * 60)
    
    batch_messages = build_batch_messages(top_cells, lidar_s2_ctx)
    
    print(f"Number of batch messages: {len(batch_messages)}")
    for j, msg in enumerate(batch_messages):
        print(f"\nMessage {j+1} role: {msg['role']}")
        if msg['role'] == 'system':
            print(f"System content length: {len(msg['content'])} chars")
            print(f"System content preview: {msg['content'][:150]}...")
        elif msg['role'] == 'user':
            print(f"User content items: {len(msg['content'])}")
            total_text_length = 0
            for k, item in enumerate(msg['content']):
                if item['type'] == 'text':
                    text_len = len(item['text'])
                    total_text_length += text_len
                    print(f"  Item {k+1}: text ({text_len} chars)")
                    # Show preview of each text section
                    if 'Regional LiDAR' in item['text']:
                        print(f"    Regional context preview: {item['text'][:100]}...")
                    elif 'Assessment Instructions' in item['text']:
                        print(f"    Instructions preview: {item['text'][:100]}...")
                    elif 'Anomaly Cells' in item['text']:
                        print(f"    Cells data preview: {item['text'][:100]}...")
                elif item['type'] == 'image_url':
                    print(f"  Item {k+1}: image_url")
            
            print(f"Total text content: {total_text_length} chars")
    
    print("\n" + "=" * 60)
    print("MESSAGE STRUCTURE COMPARISON")
    print("=" * 60)
    
    # Calculate individual vs batch efficiency
    individual_total_chars = 0
    for cell in top_cells:
        messages = build_messages(cell, lidar_s2_ctx)
        for msg in messages:
            if msg['role'] == 'user':
                for item in msg['content']:
                    if item['type'] == 'text':
                        individual_total_chars += len(item['text'])
    
    batch_total_chars = 0
    for msg in batch_messages:
        if msg['role'] == 'user':
            for item in msg['content']:
                if item['type'] == 'text':
                    batch_total_chars += len(item['text'])
    
    print(f"Individual approach total chars: {individual_total_chars}")
    print(f"Batch approach total chars: {batch_total_chars}")
    print(f"Efficiency gain: {((individual_total_chars - batch_total_chars) / individual_total_chars * 100):.1f}% reduction")
    
    # Show sample cell data structure
    print("\n" + "=" * 60)
    print("SAMPLE CELL DATA STRUCTURE")
    print("=" * 60)
    
    sample_cell = top_cells[0]
    print(f"Cell ID: {sample_cell.get('h3_cell', 'unknown')}")
    print(f"Cell features ({len(sample_cell)} total):")
    
    # Show key features
    key_features = ['mean_canopy_height', 'height_variability_norm', 'terrain_complexity_norm', 
                   'deforest_impact_norm', 'score']
    for feature in key_features:
        if feature in sample_cell:
            print(f"  {feature}: {sample_cell[feature]}")
    
    print("\nCell data structure (JSON):")
    print(json.dumps(sample_cell, indent=2)[:500] + "...")

if __name__ == "__main__":
    test_message_structure() 