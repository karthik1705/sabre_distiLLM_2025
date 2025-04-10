#%%
# Importing the necessary libraries
import pandas as pd
import numpy as np
import torch
import transformers
import accelerate
import datasets
import evaluate
import numpy
import pandas
import tqdm
import wandb
import sentencepiece
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# %%
# Raw text data
dataset_raw = pd.read_csv('Data/Product_Normalization_GRI.csv')
dataset_raw.head();

# %%
# Description column in Raw Data
raw_text = dataset_raw['Room Description'].to_list()
#raw_text_str = "; ".join(raw_text)
#raw_text_str[:200]
raw_text_str1 = raw_text[0]
#raw_text_str1

sub_raw_text = raw_text[:5] + raw_text[10000:10005]
sub_raw_text

# %%
# Normalized Product Attribute Data
normalized_product_attributes = pd.read_excel('Data/Normalized_product_attribute_name.xlsx', sheet_name = 'Normalized Product Attributes')
normalized_product_attributes.head()

# %%
# Preparing the Normalized Product Attribute data to be used for the prompt
bed_types = normalized_product_attributes['BedType'].dropna().unique()
bed_types_list = bed_types.tolist()
bed_types_str = ", ".join(bed_types_list)
bed_types_str

# %%
room_types = normalized_product_attributes['RoomType'].dropna().unique()
room_types_list = room_types.tolist()
room_types_str = ", ".join(room_types_list)
room_types_str

# %%
rate_plan_inclusives_raw = normalized_product_attributes['Rateplan Incl/Value-adds'].dropna().unique()
rate_plan_inclusives = rate_plan_inclusives_raw.tolist()
rate_plan_inclusives_str = ", ".join(rate_plan_inclusives)
rate_plan_inclusives_str

# %%
room_amenities_raw = normalized_product_attributes['Room Amenities'].dropna().unique()
room_amenities_list = room_amenities_raw.tolist()
room_amenities_str = ", ".join(room_amenities_list)
room_amenities_str

# %%
meal_plan_raw = normalized_product_attributes['Mealplan'].dropna().unique()
meal_plan_list = meal_plan_raw.tolist()
meal_plan_str = ", ".join(meal_plan_list)
meal_plan_str

# %%
room_view_raw = normalized_product_attributes['RoomView'].dropna().unique()
room_view_list = room_view_raw.tolist()
room_view_str = ", ".join(room_view_list)
room_view_str

# %%
taxes_fees_raw = normalized_product_attributes['Taxes & Fees'].dropna().unique()
taxes_fees_list = taxes_fees_raw.tolist()
taxes_fees_str = ", ".join(taxes_fees_list)
taxes_fees_str

# %%
# Fetching the model
project_id = 'fluent-vortex-449308-f6'
model_name = 'gemini-1.5-flash-002'

# %%
# Initialize the Vertex AI client
vertexai.init(project = project_id, location = 'us-central1')
model = GenerativeModel(model_name = model_name)

# %%
from vertexai.generative_models import (
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
)

# Safety Settings
safety_config = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
]

# %%
# Examples v2

examples = [
    """Text: IN THE ART OF NEW YORK|MADISON KING RM. 450 SQFT. MADISON AVE VIEWS LRGE MARBLE BATH SEP SHOWER. COMP WIFI
    JSON: {"roomType": "King Bedroom", "bedType": "Unknown", "ratePlanInclusives": ["Private bath or shower", "Complimentary wireless internet"], "roomAmenities": ["Separate tub and shower", "Wireless internet connection"], "mealPlan": "Unknown", "roomView": "Avenue View", "taxesAndFees": "Unknown"}""",

    """Text: AAA MEMBER RATE|ECO-FICIENT QUEEN ROOM RUNWAY VIEW. COMPLEMENTARY WIFI, IN ROOM COFFEE, IRON BOARD
    JSON: {"roomType": "Queen Bedroom", "bedType": "Unknown", "ratePlanInclusives": ["Eco Friendly", "Complimentary in-room coffee or tea", "Ironing board"], "roomAmenities": ["Wireless internet connection", "Ironing board"], "mealPlan": "Unknown", "roomView": "Runway View", "taxesAndFees": "Unknown"}"""
]

output_format = """
    JSON: {"roomType": "string", "bedType": "string", "ratePlanInclusives": ["string"], "roomAmenities": ["string"], "mealPlan": "string", "roomView": "string", "taxesAndFees": "string"}
"""

output_format_example = """
    JSON: {"roomType": "Queen Bedroom", "bedType": "King Bed", "ratePlanInclusives": {"values": ["Unknown"], "confidence": 0.85}, "roomAmenities": {"values": ["Wireless internet connection", "Ironing board"], "confidence": 0.85}, "mealPlan": {"value": "Unknown", "confidence": 0.8}, "roomView": {"value": "Ocean view", "confidence": 0.95}, "taxesAndFees": {"value": "Unknown", "confidence": 0.95}}
"""

# Examples
# %%

# Function to prompt the model with a single text
def process_single_text(model, text, examples, output_format, output_format_example):
    # Preparing the prompt
    prompt = f"""
    Extract the following attributes from the text below:
    - Room Type
    - Bed Type
    - Rate Plan Inclusives
    - Room Amenities
    - Meal Plan
    - Room View
    - Taxes and Fees

    Format your response as a JSON object with these exact keys: {output_format}
    
    Here are two examples of correct extractions: {examples}   

    Now extract from this text: {text} 

    Extract room attributes using ONLY the following normalized values:
    Room Types: {room_types_str}
    Bed Types: {bed_types_str}
    Rate Plan Inclusives: {rate_plan_inclusives_str}
    Room Amenities: {room_amenities_str}
    Meal Plans: {meal_plan_str}
    Room Views: {room_view_str}
    Taxes and Fees: {taxes_fees_str}

    If no clear match exists, return 'Unknown'.

    Rules for extraction:
    1. Each room must have exactly one room type
    2. Each room must have at least one bed type
    3. Rate Plan Inclusives should be a list
    4. Meal Plan is optional
    5. View is optional
    6. Amenities should be a list
    7. Do not infer attributes that aren't explicitly mentioned

    Special cases to handle:
    - If multiple bed types are mentioned, list all of them
    - If amenities are described but not in our standard list, choose the closest match
    - Unclear if ambiguous.
    - Ignore pricing and promotional information

    For each extracted attribute, provide a confidence score, like below example: {output_format_example}
    
    Follow these steps:
    1. First, identify all mentioned attributes in the text
    2. Then, match each attribute to the closest normalized value
    3. Validate against the allowed values list
    4. Format the final output as JSON without escaping
    5. Explain your reasoning for any uncertain matches

    Show your work:
    Identified attributes: ...
    Normalized matches: ...
    Final output: ...
    Reasoning: ...
    Return the final result similar to the example, with no extra text.

    """
    response = model.generate_content(prompt, safety_settings=safety_config)
    return response

#%%

#%%
# Running for 1 input
import json

result_single = process_single_text(model, raw_text_str1, examples, output_format, output_format_example)

with open('result_single.json', 'w') as file:
    json.dump(result_single.text, file, indent=4)

# %%
# Running for multiple input texts
# Process texts with the model in a loop
results_few = []
for text in sub_raw_text:
    try:
        result = process_single_text(model, text, examples,  output_format, output_format_example)
        results_few.append(result)
    except Exception as e:
        print(f"Error processing text: {text[:50]}...")
        print(f"Error: {str(e)}")
        results_few.append(None)

#%%
import json
with open('results_few.json', 'w') as file:
    for res in results_few:
        json.dump(res.text, file, indent=4)

# %%
import json
with open('results_few.json', 'w') as file:
    json.dump(results_few, file, indent=4)

#results.save('results.json')


# %%
