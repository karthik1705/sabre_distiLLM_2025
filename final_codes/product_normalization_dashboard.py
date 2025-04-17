import pandas as pd
import numpy as np
import streamlit as st

# Header Section
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

# Input Section
with st.form("description_form"):
    hotel_description = st.text_area("📝 Room Description", height=200, placeholder="E.g., A deluxe king bedroom with sea view and breakfast included...")
    submitted = st.form_submit_button("Classify")

# Detailed Input & Output
if submitted:
    if hotel_description.strip() == "":
        st.warning("Please enter a valid description.")
    else:
        # Placeholder(switch to llm result)
        st.markdown("### 🛏️ Extracted Room Type:")
        st.success("Deluxe Room")

# Footer Section
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 14px;'>
        Created by <b>UT Austin & Sabre Team</b> | Template for room type classification
    </p>
""", unsafe_allow_html=True)
