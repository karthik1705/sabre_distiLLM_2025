# Attribute Extraction & Hotel Product Normalization Using LLMs

- Recommended: Use a virtual environment (e.g., venv, conda) to isolate dependencies
- Use VS Code / Cursor or tools that have capability to open .ipynb files
- Install requirements from the requirements.txt file with this command:

    ```bash
    pip install -r requirements.txt
    ```

# 1. Code Overview
# A. Data Preprocessing & EDA (dataset_preprocess.ipynb & EDA.ipynb)
- Short words in descriptions were mapped to full words
- Duplicates were removed
- 1 Missing label was filled with the potential Room Type (Bungalow)
- Processed file was stored as Product_Normalization_GRI_Expanded.csv in 'Data', which will be used for all the experiments
- Dataset characteristics & Frequencies of words/values were analyzed

# B. Supervised Models (supervised_models.ipynb)
- Stratified Samples of sizes 1%, 5%, 10% were created for finetuning
- Prompt Engineering and Finetuning methods are hosted in this file
- Finetuned LLAMA model is in the finetuned_llama_3_1b.ipynb file

Credentials Required:
- Hugging Face Token (Generate from https://huggingface.co/)
- GEMINI - Vertex Project ID (Create at https://cloud.google.com/vertex-ai)
- LLAMA/BERT Model Accesses - Navigate to a specific model's page on HuggingFace and request for access (https://huggingface.co/)
- Weights & Biases (wandb) credentials to store logs of experiments - (Create on https://wandb.ai/site/)

# C. Unsupervised Models (unsupervised_models.ipynb)
- All unsupervised methods mentioned in the report are in this file
- Semi-supervised clustering method using SentenceTransformer embeddings



# 2. Product Normalization Dashboard
- Goal: Extract Room Type from a Hotel Room Description using a Finetuned BERT-base-uncased model
- Input: Room Description
- Outputs: Extracted Room Type and Actual Room Type (if any)

Before running the dashboard:
- Make sure the requirements are installed from requirements.txt as mentioned above.
- Make sure that the Product_Normalization_GRI_Expanded.csv dataset is inside 'Data' folder.
- Verify that the BERT model sits in the folder 'final_bert_base_uncased'.

- To run the dashboard, run the following code in the terminal:

    ```bash
    streamlit run '../final_codes/product_normalization_dashboard.py'
    ```