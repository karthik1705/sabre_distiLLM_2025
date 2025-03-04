#%% Importing libraries
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import pandas as pd
import random
from tqdm import tqdm

from huggingface_hub import login
login(token='')

#%% Load and prepare data
dataset_raw = pd.read_csv('Data/Product_Normalization_GRI.csv')
normalized_product_attributes = pd.read_excel('Data/Normalized_product_attribute_name.xlsx', 
                                            sheet_name='Normalized Product Attributes')

# Get room types
room_types = normalized_product_attributes['RoomType'].dropna().unique()
room_types_list = room_types.tolist()
room_types_str = ", ".join(room_types_list)

#%% Setup Device
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU only")
    return device

#%% Setup Llama Pipeline
def setup_llama_pipeline():
    model_id = "meta-llama/Llama-3.2-1B"  # Changed from 2 7B model to 3.2 1B model
    
    device = setup_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    return pipeline, tokenizer

#%% Process Single Text
def process_single_text(text, pipeline, tokenizer, room_types_str):
    # Check for accessibility keywords
    is_accessible = 'access' in text.lower() or 'ada' in text.lower()
    
    old_prompt = f"""Extract room type from this text: {text}
    Use ONLY these normalized values: {room_types_str}
    Rules:
    1. Match to closest normalized value
    2. Use "Unknown" if unsure
    3. If text contains 'ACCESS' or 'ADA', classify as 'Accessible Room'
    4. Score confidence between 0.0 and 1.0
    
    Format output as: room_type, confidence_score
    """
    
    prompt = f"""Given this hotel room description: {text}

Available room types are: {room_types_str}

If the description contains 'ACCESS' or 'ADA', output exactly: Accessible Room, 1.0

Otherwise, identify the most appropriate room type from the available options.
Output your answer in exactly this format:
<room type>, <confidence score between 0.0 and 1.0>

Output:"""
    
    # If accessible, return immediately
    if is_accessible:
        return {
            "roomType": "Accessible Room",
            "confidence": 1.0
        }
    
    outputs = pipeline(
        prompt,
        max_new_tokens=64,  # Reduced tokens from 128 since we expect short response
        temperature=0.3,    # Reduced temperature from 0.7 for more focused outputs
        top_p=0.9,          # Reduced from 0.95
        top_k=10,           # Reduced from 40 for more focused selection
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract response
    response_text = outputs[0]['generated_text']
    
    # Parse comma-separated values
    try:
        room_type, confidence = response_text.strip().split(',')
        return {
            "roomType": room_type.strip(),
            "confidence": float(confidence.strip())
        }
    except ValueError:
        return {
            "roomType": "Unknown",
            "confidence": 0.0
        }

#%% Process Batch
def process_batch(texts, pipeline, tokenizer, room_types_str):
    results = []
    for text in tqdm(texts, desc="Processing texts"):
        result = process_single_text(text, pipeline, tokenizer, room_types_str)
        results.append(result)
    return results

#%% Evaluate Results
def evaluate_llama_accuracy(results, true_labels):
    correct_top = 0
    total = len(results)
    
    print("\nLlama Extraction Evaluation:")
    print("===========================")
    
    for i, (result, true_label) in enumerate(zip(results, true_labels)):
        pred_type = result["roomType"].lower()
        true_label = str(true_label).lower()
        confidence = result["confidence"]
        
        # Check match
        match = true_label in pred_type or pred_type in true_label
        correct_top += match
        
        # Print first 5 examples
        if i < 5:
            print(f"\nTrue Label: {true_label}")
            print(f"Predicted: {pred_type} (confidence: {confidence:.3f})")
            print(f"Match: {'✓' if match else '✗'}")
    
    accuracy = correct_top / total
    print(f"\nAccuracy: {accuracy:.2%}")
    return accuracy

#%% Main Execution
if __name__ == "__main__":
    # Setup
    pipeline, tokenizer = setup_llama_pipeline()
    
    # Sample data for testing
    sample_size = 100
    sampled_data = dataset_raw.sample(n=sample_size, random_state=42)
    
    # Process samples
    results = process_batch(
        sampled_data['Room Description'].tolist(),
        pipeline,
        tokenizer,
        room_types_str
    )
    
    # Evaluate
    accuracy = evaluate_llama_accuracy(results, sampled_data['Guest Room Info'])
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df['Description'] = sampled_data['Room Description'].values
    output_df['True_Label'] = sampled_data['Guest Room Info'].values
    output_df.to_csv('llama_results_3_2_1B_modified.csv', index=False)
# %%
