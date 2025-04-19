import streamlit as st
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import eeg_to_spectrogram
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
import gdown

# Workaround for Lambda layers used as SlicingOpLambda
class SlicingOpLambda(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(SlicingOpLambda, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return inputs  # Pass-through identity (since actual slicing logic was lost in saved model)

# Register the custom dropout and Lambda layer
class FixedDropout(Dropout):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)
        self.supports_masking = True
    
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
        return super(FixedDropout, self).call(inputs, training)

# Register custom objects for TensorFlow
get_custom_objects().update({
    'FixedDropout': FixedDropout,
    'SlicingOpLambda': SlicingOpLambda  # Register the fixed Lambda layer
})

# Google Drive IDs for models
drive_links = [
    "19vagTsjJushCJ25YikZzkCTyaLFfmfO-",
    "1LhptLaTjdDQ7KAoKzYCgUqNrvDFdOyci",
    "1iYXG31bFpLT-eIIFCk7qLSKnd67kwUP8",
    "1e7AEIA2sdJid1T5_HVDfTZz2NzWGYVhZ",
    "13KoESOQzPG1GwaFD5BBRT-SudBhkMD-k"
]

# Download and load models
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    models = []
    for i, file_id in enumerate(drive_links):
        model_path = f"models/EffNetB0_Fold{i}.h5"
        if not os.path.exists(model_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'FixedDropout': FixedDropout, 'SlicingOpLambda': SlicingOpLambda},
            compile=False  # We don't need to compile the model for inference
        )
        models.append(model)
    return models

# EEG class descriptions
activity_descriptions = {
    'Seizure': "Seizure activity is characterized by sudden and uncontrolled electrical disturbances in the brain...",
    'LPD': "LPD (Localize Paroxysmal Discharge)...",
    'GPD': "GPD (Generalized Paroxysmal Discharge)...",
    'LRDA': "LRDA (Low-Risk Developmental Abnormality)...",
    'GRDA': "GRDA (Generalized Risk Developmental Abnormality)...",
    'Other': "Other includes any brain activity that doesn't fit into the predefined categories..."
}

# User input
name = st.text_input("Please enter your name:")
uploaded_file = st.file_uploader("📁 Upload EEG .parquet File", type=["parquet"])

if uploaded_file and name:
    st.success(f"EEG file uploaded. Processing for {name}...")
    try:
        df = pd.read_parquet(uploaded_file)
        st.write("📋 EEG Columns Found:", df.columns.tolist())
        spec = eeg_to_spectrogram(df)
        x = np.zeros((1, 128, 256, 8), dtype='float32')
        for i in range(4):
            x[0,:,:,i] = spec[:,:,i]
            x[0,:,:,i+4] = spec[:,:,i]
        models = load_models()
        preds = [model.predict(x)[0] for model in models]
        final_pred = np.mean(preds, axis=0)
        labels = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        max_idx = np.argmax(final_pred)
        max_label = labels[max_idx]
        max_val = final_pred[max_idx]

        st.markdown("### 📊 Prediction Probabilities")
        col1, col2 = st.columns(2)
        for i, label in enumerate(labels):
            with col1 if i % 2 == 0 else col2:
                color = "#1565c0" if label == max_label else "#455a64"
                st.markdown(f"<div style='color:{color}; font-weight:bold'>{label}: {final_pred[i]:.4f}</div>", unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({'Class': labels, 'Probability': final_pred}).set_index('Class'))

        st.markdown("### 📝 Diagnosis Summary")
        st.markdown(f"<div class='diagnosis-result'>🧠 Most Likely: <b>{max_label} ({max_val:.4f})</b></div>", unsafe_allow_html=True)
        st.markdown("💡 This result represents the most probable type of harmful brain activity detected.")
        st.markdown("### 📚 Activity Description")
        st.markdown(f"<div style='font-size: 16px;'>{max_label}: {activity_descriptions[max_label]}</div>", unsafe_allow_html=True)

        report = f"""
        <h2 style="text-align: center; color: #1565c0;">Brain EEG Diagnosis Report</h2>
        <p><strong>Patient Name:</strong> {name}</p>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Diagnosis:</strong> Most Likely: <strong>{max_label}</strong> ({max_val:.4f})</p>
        <h4>Activity Description:</h4>
        <p>{activity_descriptions[max_label]}</p>
        """
        st.markdown("### 📝 Download Diagnosis Report")
        st.download_button("📥 Download Medical Report", report, "diagnosis_report.html", "text/html")

    except Exception as e:
        st.error(f"❌ Error: {e}")
