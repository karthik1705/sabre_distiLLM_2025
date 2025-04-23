import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

# 2) Loading the fine‑tuned BERT model & tokenizer (cached so it only happens once)
@st.cache_resource
def load_model(path="final_bert_base_uncased"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model("final_bert_base_uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3) Grab the id2label mapping from the model’s config
id2label = {int(k):v for k,v in model.config.id2label.items()}

# 4) Define a helper to run inference
def predict_room_type(text: str) -> str:
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(logits.argmax(-1))
    return id2label[pred_id]

# 5) Input form
with st.form("description_form"):
    hotel_description = st.text_area(
        "📝 Room Description",
        height=200,
        placeholder="E.g., A deluxe king bedroom with sea view and breakfast included..."
    )
    submitted = st.form_submit_button("Classify")

# 6) On submit, run your model and display the result
if submitted:
    if not hotel_description.strip():
        st.warning("Please enter a valid description.")
    else:
        room_type = predict_room_type(hotel_description)
        st.markdown("### 🛏️ Extracted Room Type:")
        st.success(room_type)

# 7) Footer
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px;'>
        Created by <b>UT Austin & Sabre Team</b> | Template for room type classification
    </p>
""", unsafe_allow_html=True)
