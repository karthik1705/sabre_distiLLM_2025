import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 0) Adjust these paths / column names to match your setup
DATA_PATH          = "../Data/Product_Normalization_GRI_Expanded.csv"
TEXT_COL           = "Room Description Expanded"
TRUE_LABEL_COL     = "Guest Room Info"
MODEL_PATH         = "final_bert_base_uncased"

# 1) Page config & header
st.set_page_config(
    page_title="Hotel Room Type Classifier",
    page_icon="🏨",
    layout="centered"
)
st.markdown("""
    <h1 style='text-align: center;'>🏨 Hotel Room Type Classifier</h1>
    <p style='text-align: center; font-size: 18px;'>
        Enter a description of a hotel room and get the extracted room type.
    </p>
    <hr>
""", unsafe_allow_html=True)

# 2) Load dataset (cached)
@st.cache_data
def load_data(path):
    return pd.read_csv(path, dtype=str)

df = load_data(DATA_PATH)

# 3) Load the fine-tuned BERT model & tokenizer (cached)
@st.cache_resource
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4) Grab the id2label mapping from the model’s config
id2label = {int(k): v for k, v in model.config.id2label.items()}

# 5) Inference helper
def predict_room_type(text: str) -> str:
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(logits.argmax(-1))
    return id2label[pred_id]

# 6) Input form
with st.form("description_form"):
    hotel_description = st.text_area(
        "📝 Room Description",
        height=200,
        placeholder="E.g., A deluxe king bedroom with sea view and breakfast included..."
    )
    submitted = st.form_submit_button("Classify")

# 7) On submit, run your model & lookup actual label
if submitted:
    if not hotel_description.strip():
        st.warning("Please enter a valid description.")
    else:
        # Model prediction
        room_type_pred = predict_room_type(hotel_description)
        st.markdown("### 🛏️ Predicted Room Type:")
        st.success(room_type_pred)

        # Lookup actual label in your DataFrame
        matches = df[df[TEXT_COL] == hotel_description]
        if len(matches) == 1:
            actual_label = matches.iloc[0][TRUE_LABEL_COL]
            st.markdown("### ✅ Actual Room Type:")
            st.info(actual_label)
        elif len(matches) > 1:
            st.warning("Multiple matching entries found in dataset.")
        else:
            st.warning("No exact match found in dataset for this description.")

# 8) Footer
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px;'>
        Created by <b>UT Austin & Sabre Team</b> | Template for room type classification
    </p>
""", unsafe_allow_html=True)
