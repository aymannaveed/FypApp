import streamlit as st
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import gdown
import os
from utils import eeg_to_spectrogram

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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🧠 Harmful Brain Activity Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="small-font">EEG Diagnosis for a Smarter Tomorrow!</p>', unsafe_allow_html=True)
st.markdown("---")

# Handle custom dropout layer
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K

class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.supports_masking = True
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        return super(FixedDropout, self).call(inputs, training)

# Custom layer to handle SlicingOpLambda error
class SlicingOpLambda(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SlicingOpLambda, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Default implementation - modify if you know the exact slicing needed
        return inputs[:, :, :, :]

# Register custom objects
get_custom_objects().update({
    'FixedDropout': FixedDropout,
    'SlicingOpLambda': SlicingOpLambda,
    'tf.operators.getitem': SlicingOpLambda  # Handle the legacy name
})

# Load all 5 models from Google Drive
@st.cache_resource
def load_models():
    models = []
    model_urls = [
        'https://drive.google.com/uc?id=19vagTsjJushCJ25YikZzkCTyaLFfmfO-',
        'https://drive.google.com/uc?id=1LhptLaTjdDQ7KAoKzYCgUqNrvDFdOyci',
        'https://drive.google.com/uc?id=1iYXG31bFpLT-eIIFCk7qLSKnd67kwUP8',
        'https://drive.google.com/uc?id=1e7AEIA2sdJid1T5_HVDfTZz2NzWGYVhZ',
        'https://drive.google.com/uc?id=13KoESOQzPG1GwaFD5BBRT-SudBhkMD-k'
    ]
    
    # Create a temporary directory for models
    os.makedirs('temp_models', exist_ok=True)
    
    for i, url in enumerate(model_urls):
        try:
            output_path = f'temp_models/EffNetB0_Fold{i}.h5'
            if not os.path.exists(output_path):
                with st.spinner(f'Downloading model {i+1}/5...'):
                    gdown.download(url, output_path, quiet=True)
            
            model = tf.keras.models.load_model(
                output_path,
                custom_objects=get_custom_objects(),
                compile=False  # Try with compile=False if issues persist
            )
            models.append(model)
        except Exception as e:
            st.error(f"Error loading model {i}: {str(e)}")
            st.error("Please ensure the model files are not corrupted.")
            raise e
    
    return models

# Define the description for each label
activity_descriptions = {
    'Seizure': "Seizure activity is characterized by sudden and uncontrolled electrical disturbances in the brain, often leading to convulsions or loss of consciousness.",
    'LPD': "LPD (Localize Paroxysmal Discharge) refers to abnormal electrical activity occurring in a specific region of the brain, often associated with certain neurological conditions.",
    'GPD': "GPD (Generalized Paroxysmal Discharge) involves widespread brain activity that can cause sudden physical or behavioral changes, commonly seen in epilepsy.",
    'LRDA': "LRDA (Low-Risk Developmental Abnormality) refers to less severe brain anomalies often observed during brain development, potentially leading to cognitive delays or learning difficulties.",
    'GRDA': "GRDA (Generalized Risk Developmental Abnormality) is a more severe form of developmental abnormality that can affect brain function and cognitive abilities.",
    'Other': "Other includes any brain activity that doesn't fit into the predefined categories, potentially indicating a variety of neurological conditions."
}

# Ask for user's name
name = st.text_input("Please enter your name:")

# File upload
uploaded_file = st.file_uploader("📁 Upload EEG .parquet File", type=["parquet"])

if uploaded_file and name:
    st.success(f"EEG file uploaded. Processing for {name}...")

    try:
        # Read and process EEG data
        df = pd.read_parquet(uploaded_file)
        st.write("📋 EEG Columns Found:", df.columns.tolist())

        # Generate spectrogram
        with st.spinner('Generating spectrogram...'):
            spec = eeg_to_spectrogram(df)

        # Format input for EfficientNet
        x = np.zeros((1, 128, 256, 8), dtype='float32')
        for i in range(4):
            x[0,:,:,i] = spec[:,:,i]
            x[0,:,:,i+4] = spec[:,:,i]

        # Load models and make predictions
        with st.spinner('Loading models and making predictions...'):
            models = load_models()
            preds = [model.predict(x, verbose=0)[0] for model in models]  # verbose=0 to suppress output
            final_pred = np.mean(preds, axis=0)

        labels = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        max_idx = np.argmax(final_pred)
        max_label = labels[max_idx]
        max_val = final_pred[max_idx]

        # Prediction Display
        st.markdown("### 📊 Prediction Probabilities")
        col1, col2 = st.columns(2)
        for i, label in enumerate(labels):
            with col1 if i % 2 == 0 else col2:
                color = "#1565c0" if label == max_label else "#455a64"
                st.markdown(f"<div style='color:{color}; font-weight:bold'>{label}: {final_pred[i]:.4f}</div>", unsafe_allow_html=True)

        results_df = pd.DataFrame({'Class': labels, 'Probability': final_pred})
        st.bar_chart(results_df.set_index('Class'))

        # Diagnosis Summary
        st.markdown("### 📝 Diagnosis Summary")
        st.markdown(f"<div class='diagnosis-result'>🧠 Most Likely: <b>{max_label} ({max_val:.4f})</b></div>", unsafe_allow_html=True)
        st.markdown("💡 This result represents the most probable type of harmful brain activity detected.")

        # Display brief description of the detected activity
        st.markdown("### 📚 Activity Description")
        st.markdown(f"<div style='font-size: 16px;'>{max_label}: {activity_descriptions[max_label]}</div>", unsafe_allow_html=True)

        # Generate medical report
        report = f"""
        <h2 style="text-align: center; color: #1565c0;">Brain EEG Diagnosis Report</h2>
        <p><strong>Patient Name:</strong> {name}</p>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Diagnosis:</strong> Most Likely: <strong>{max_label}</strong> ({max_val:.4f})</p>
        <h4>Activity Description:</h4>
        <p>{activity_descriptions[max_label]}</p>
        """

        # Downloadable report
        st.markdown("### 📝 Download Diagnosis Report")
        st.download_button("📥 Download Medical Report", report, "diagnosis_report.html", "text/html")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error("If the error persists, please check your input file and try again.")
