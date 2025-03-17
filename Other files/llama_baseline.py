#%% Importing libraries
# Authenticating with Hugging Face (Terminal Login so below is not needed)
#os.environ['HF_TOKEN'] = ''
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import pandas as pd
import random
# %%
# Raw text data
dataset_raw = pd.read_csv('Data/Product_Normalization_GRI.csv')
dataset_raw.head()

# %%
# Description column in Raw Data
raw_text = dataset_raw['Room Description'].to_list()
#raw_text_str1 = raw_text[0]
#raw_text_str1
#sub_raw_text = raw_text[:5] + raw_text[10000:10005]
#sub_raw_text
random_raw_text = random.sample(raw_text, 20)
random_raw_text

with open('input_random_raw_texts_0206_llama_8b_baseline.txt', 'w') as file:
    for text in random_raw_text:
        file.write(text + '\n')

# %%
# Normalized Product Attribute Data
normalized_product_attributes = pd.read_excel('Data/Normalized_product_attribute_name.xlsx', sheet_name = 'Normalized Product Attributes')
normalized_product_attributes.head()

# %%
room_types = normalized_product_attributes['RoomType'].dropna().unique()
room_types_list = room_types.tolist()
room_types_str = ", ".join(room_types_list)
room_types_str

# %%
# Examples

examples = [
    """Text: IN THE ART OF NEW YORK|MADISON KING RM. 450 SQFT. MADISON AVE VIEWS LRGE MARBLE BATH SEP SHOWER. COMP WIFI
    JSON: {"roomType": "King Bedroom", "confidence": 0.82}""",

    """Text: AAA MEMBER RATE|ECO-FICIENT QUEEN ROOM RUNWAY VIEW. COMPLEMENTARY WIFI, IN ROOM COFFEE, IRON BOARD
    JSON: {"roomType": "Queen Bedroom", "confidence": 0.95}"""
]

output_format = """
    JSON: {"roomType": "string", "confidence": float}
"""

#%%
# Setup Device
def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M-series GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU only")
    return device

# %%
# Setup Llama Pipeline
def setup_llama_pipeline():
    # Model initialization
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    device = setup_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Move model to MPS device if available
    if device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": device}  # Map all modules to MPS
        )
    else:
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

# %%
# Process Single Text Function
def process_single_text(text, pipeline, tokenizer, examples, output_format, room_types_str):
    prompt = f"""<s>[INST] You are a helpful assistant that extracts and normalizes room attributes.

    Extract "Room Type" from this text, along with a confidence score: {text}

    Extract Room Type using ONLY the normalized values from this list: {room_types_str}
    Rules for extraction:
    1. If no clear match exists, return 'Unknown'.
    2. Each room must have exactly one room type

    Return a JSON object with these exact keys and confidence scores: {output_format}

    Here are two examples of correct extractions: {examples}

    Rules:
    1. Match to closest normalized value
    2. Use "Unknown" if unsure
    3. Don't infer missing attributes
    4. Score confidence between 0.0 and 1.0 [/INST]

    Let me analyze this step by step:
    1. First, I'll identify the Room Type attribute
    2. Then, I'll match it to normalized value
    3. Finally, I'll format the JSON response

    Here's the extracted and normalized data:</s>
    """

    prompt_modified = f"""
    Extract "Room Type" attribute from the TEXT below, along with a confidence score:
    Follow these steps:
    1. First, remember all the room types in the normalized list: {room_types_str}
    2. For TEXT, observe carefully and identify which is the closest normalized value.
    3. Validate against the allowed values list.
    4. When it is ambiguous, return the most likely room type and low confidence score accordingly.
    5. Very rarely there might be two room types during which case return both room types with appropriate confidence scores.
    6. If no match is found, return 'Unknown' and very low confidence score.
    7. Format the final output as plain text with roomType and confidence parameters separated by commas.
    
    Now extract from this TEXT: {text}

    """

    # Generate response
    outputs = pipeline(
        prompt_modified,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
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
        # Return Unknown with 0 confidence if parsing fails
        return {
            "roomType": "Unknown",
            "confidence": 0.0
        }

# %%
# Running for 1 input
pipeline, tokenizer = setup_llama_pipeline()
result = process_single_text(random_raw_text[0], pipeline, tokenizer, examples, output_format, room_types_str)
result


# %%
