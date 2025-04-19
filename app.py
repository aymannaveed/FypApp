import streamlit as st
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os
from keras.utils import get_custom_objects
from utils import eeg_to_spectrogram

# Remove custom layer registration, allowing TensorFlow to handle layers like 'getitem' automatically

# Set the page config
st.set_page_config(page_title="Brain EEG Classifier", layout="wide")

# Styling (medical blue aesthetic)
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
        .stSpinner > div {
            text-align: center;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🧠 Harmful Brain Activity Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">EEG Diagnosis for a Smarter Tomorrow!</p>', unsafe_allow_html=True)
st.markdown("---")

# Activity Descriptions
activity_descriptions = {
    'Seizure': "Seizure activity is characterized by sudden and uncontrolled electrical disturbances in the brain.",
    'LPD': "LPD (Localize Paroxysmal Discharge) refers to abnormal electrical activity in a specific brain region.",
    'GPD': "GPD (Generalized Paroxysmal Discharge) involves widespread brain activity changes.",
    'LRDA': "LRDA (Low-Risk Developmental Abnormality) refers to less severe brain anomalies.",
    'GRDA': "GRDA (Generalized Risk Developmental Abnormality) affects brain function and cognition.",
    'Other': "Other includes activity that doesn't fit predefined categories."
}

# Model Loading Function
@st.cache_resource
def load_models():
    model_files = {
        'EffNetB0_Fold0.h5': 'https://drive.google.com/uc?id=19vagTsjJushCJ25YikZzkCTyaLFfmfO-',
        'EffNetB0_Fold1.h5': 'https://drive.google.com/uc?id=1LhptLaTjdDQ7KAoKzYCgUqNrvDFdOyci',
        'EffNetB0_Fold2.h5': 'https://drive.google.com/uc?id=1iYXG31bFpLT-eIIFCk7qLSKnd67kwUP8',
        'EffNetB0_Fold3.h5': 'https://drive.google.com/uc?id=1e7AEIA2sdJid1T5_HVDfTZz2NzWGYVhZ',
        'EffNetB0_Fold4.h5': 'https://drive.google.com/uc?id=13KoESOQzPG1GwaFD5BBRT-SudBhkMD-k'
    }
    
    os.makedirs('models', exist_ok=True)
    models = []
    
    for filename, url in model_files.items():
        model_path = os.path.join('models', filename)
        try:
            if not os.path.exists(model_path):
                with st.spinner(f'Downloading {filename}...'):
                    gdown.download(url, model_path, quiet=True)
            
            with st.spinner(f'Loading {filename}...'):
                model = tf.keras.models.load_model(model_path, compile=False)
                models.append(model)
        except Exception as e:
            st.error(f"Failed to load {filename}: {str(e)}")
            st.error("Please check your internet connection and try again.")
            raise
    
    return models

# Main App
def main():
    name = st.text_input("Please enter your name:")
    uploaded_file = st.file_uploader("📁 Upload EEG .parquet File", type=["parquet"])
    
    if uploaded_file and name:
        try:
            # Load data
            with st.spinner('Reading EEG data...'):
                df = pd.read_parquet(uploaded_file)
            
            # Process data
            with st.spinner('Processing spectrogram...'):
                spec = eeg_to_spectrogram(df)
                x = np.zeros((1, 128, 256, 8), dtype='float32')
                for i in range(4):
                    x[0,:,:,i] = spec[:,:,i]
                    x[0,:,:,i+4] = spec[:,:,i]
            
            # Load models and predict
            with st.spinner('Initializing models...'):
                models = load_models()
            
            with st.spinner('Making predictions...'):
                preds = [model.predict(x, verbose=0)[0] for model in models]
                final_pred = np.mean(preds, axis=0)
            
            # Display results
            labels = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
            max_idx = np.argmax(final_pred)
            max_label = labels[max_idx]
            max_prob = final_pred[max_idx]
            
            st.markdown("### 📊 Prediction Results")
            cols = st.columns(2)
            for i, (label, prob) in enumerate(zip(labels, final_pred)):
                with cols[i % 2]:
                    st.metric(label, f"{prob:.2%}", 
                            delta="HIGHEST" if label == max_label else None)
            
            st.markdown("### 📝 Diagnosis Summary")
            st.success(f"Most likely activity: **{max_label}** ({max_prob:.2%} confidence)")
            st.markdown(f"**Description:** {activity_descriptions[max_label]}")
            
            # Generate report
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
            st.error("Please check your input file and try again.")

if __name__ == "__main__":
    main()
