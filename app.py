import streamlit as st
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os
from utils import eeg_to_spectrogram

# Page setup
st.set_page_config(page_title="Brain EEG Classifier", layout="wide")

# UI styles
st.markdown("""
    <style>
        .stApp { background-color: #e3f2fd; }
        .big-font {
            font-size: 35px !important;
            font-weight: 600;
            color: #1565c0;
            text-align: center;
        }
        .small-font {
            font-size: 18px !important;
            color: #455a64;
            text-align: center;
        }
        .stButton>button {
            background-color: #1565c0;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-size: 16px;
        }
        .stDownloadButton>button {
            background-color: #1e88e5;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🧠 Harmful Brain Activity Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">EEG Diagnosis for a Smarter Tomorrow!</p>', unsafe_allow_html=True)
st.markdown("---")

# Model URLs (Google Drive direct links via gdown)
model_urls = [
    'https://drive.google.com/uc?id=19oM31bN9Az-ZQg26RLKoMHreJUxBnE-0',
    'https://drive.google.com/uc?id=15OKbJj7x6ffHQ7DrY7Yima3p0JpQEAug',
    'https://drive.google.com/uc?id=1bsVQOnGax39tpVxl2N6bLWx1MfdnvanZ',
    'https://drive.google.com/uc?id=1BMJlP_Q1kyqMyxm2iXU6xuD1nMirvzAS',
    'https://drive.google.com/uc?id=1pgh3q1RStFeoegSZfL1HjB-GUUl7ehi6',
]

# Labels
labels = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

# Class descriptions
activity_descriptions = {
    'Seizure': "Seizure activity is characterized by sudden and uncontrolled electrical disturbances in the brain.",
    'LPD': "LPD (Localize Paroxysmal Discharge) refers to abnormal electrical activity in a specific brain region.",
    'GPD': "GPD (Generalized Paroxysmal Discharge) involves widespread brain activity changes.",
    'LRDA': "LRDA (Low-Risk Developmental Abnormality) refers to less severe brain anomalies.",
    'GRDA': "GRDA (Generalized Risk Developmental Abnormality) affects brain function and cognition.",
    'Other': "Other includes activity that doesn't fit predefined categories."
}

# Load models with cache
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    models = []
    for i, url in enumerate(model_urls):
        filename = f"EffNetB0_Fold{i}.h5"
        filepath = os.path.join("models", filename)
        if not os.path.exists(filepath):
            gdown.download(url, filepath, quiet=False)
        model = tf.keras.models.load_model(filepath, compile=False)
        models.append(model)
    return models

# Main app logic
def main():
    name = st.text_input("Please enter your name:")
    uploaded_file = st.file_uploader("📁 Upload EEG CSV", type=["csv", "parquet"])

    if uploaded_file and name:
        try:
            # Read file
            with st.spinner('Reading EEG data...'):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_parquet(uploaded_file)
            
            # Spectrogram
            with st.spinner('Converting to spectrogram...'):
                spec = eeg_to_spectrogram(df)
                x = np.zeros((1, 128, 256, 8), dtype='float32')
                for i in range(4):
                    x[0, :, :, i] = spec[:, :, i]
                    x[0, :, :, i+4] = spec[:, :, i]

            # Load models and predict
            with st.spinner("Loading models and predicting..."):
                models = load_models()
                preds = [model.predict(x, verbose=0)[0] for model in models]
                final_pred = np.mean(preds, axis=0)
                max_idx = np.argmax(final_pred)
                max_label = labels[max_idx]
                max_prob = final_pred[max_idx]

            # Results
            st.markdown("### 📊 Prediction Results")
            cols = st.columns(2)
            for i, (label, prob) in enumerate(zip(labels, final_pred)):
                with cols[i % 2]:
                    st.metric(label, f"{prob:.2%}", delta="HIGHEST" if label == max_label else None)

            st.markdown("### 📝 Diagnosis Summary")
            st.success(f"Most likely activity: **{max_label}** ({max_prob:.2%} confidence)")
            st.markdown(f"**Description:** {activity_descriptions[max_label]}")

            # Downloadable report
            report = f"""
            <h2>Brain EEG Diagnosis Report</h2>
            <p><strong>Patient:</strong> {name}</p>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <p><strong>Diagnosis:</strong> {max_label} ({max_prob:.2%})</p>
            <p><strong>Details:</strong> {activity_descriptions[max_label]}</p>
            """
            st.download_button("📥 Download Report", report, "eeg_report.html", "text/html")

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            st.info("Please ensure the EEG file format is correct and retry.")

if __name__ == "__main__":
    main()
