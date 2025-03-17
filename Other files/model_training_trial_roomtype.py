#%%
# Importing the necessary libraries
import pandas as pd
import numpy as np
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel
import random
# %%
# Raw text data
dataset_raw = pd.read_csv('Data/Product_Normalization_GRI.csv')
dataset_raw.head();

# %%
# Description column in Raw Data
raw_text = dataset_raw['Room Description'].to_list()
#raw_text_str1 = raw_text[0]
#raw_text_str1

#sub_raw_text = raw_text[:5] + raw_text[10000:10005]
#sub_raw_text

random_raw_text_0206 = random.sample(raw_text, 100)
random_raw_text_0206

with open('input_random_raw_texts_0206_100.txt', 'w') as file:
    for text in random_raw_text_0206:
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
# Fetching the model
project_id = 'fluent-vortex-449308-f6'
model_name = 'gemini-1.5-flash-002'

# %%
# Initialize the Vertex AI client
vertexai.init(project = project_id, location = 'us-central1')
model = GenerativeModel(model_name = model_name)

# %%
# Examples v2

examples = [
    """Text: IN THE ART OF NEW YORK|MADISON KING RM. 450 SQFT. MADISON AVE VIEWS LRGE MARBLE BATH SEP SHOWER. COMP WIFI
    JSON: {"roomType": "King Bedroom", "confidence": 0.82}""",

    """Text: AAA MEMBER RATE|ECO-FICIENT QUEEN ROOM RUNWAY VIEW. COMPLEMENTARY WIFI, IN ROOM COFFEE, IRON BOARD
    JSON: {"roomType": "Queen Bedroom", "confidence": 0.95}"""
]

output_format = """
    JSON: {"roomType": "string", "confidence": float}
"""


# %%

# Function to prompt the model with a single text
def process_single_text(model, text, examples,  output_format):
    # Preparing the prompt
    prompt = f"""
    Extract "Room Type" from the text below, along with a confidence score:

    Format your response as a JSON object with these exact keys: {output_format}
    
    Here are two examples of correct extractions: {examples}   

    Now extract from this text: {text} 

    Extract Room Type using ONLY the normalized values from this list: {room_types_str}
    Rules for extraction:
    1. If no clear match exists, return 'Unknown'.
    2. Each room must have exactly one room type

    Follow these steps:
    1. First, identify room type in the text
    2. Then, match it to the closest normalized value
    3. Validate against the allowed values list
    4. Format the final output as JSON

    Return the final result similar to the example, with no extra text.

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


    response = model.generate_content(prompt_modified)
    return response


# %%
# Running for multiple input texts
# Process texts with the model in a loop
results_random_0206 = []
for text in random_raw_text_0206:
    try:
        result = process_single_text(model, text, examples,  output_format)
        results_random_0206.append(result)
    except Exception as e:
        print(f"Error processing text: {text[:50]}...")
        print(f"Error: {str(e)}")
        results_random_0206.append(None)

#%%
# Saving input

with open('results_random_0206_no_examples_and_modified_prompt_100.txt', 'w') as file:
    for res in results_random_0206:
        file.write(res.text)
# %%
#results_random_0206
# %%
