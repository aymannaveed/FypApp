import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import requests
import os

from tensorflow.keras.models import load_model
from io import BytesIO

# --- Custom Layer Fix ---
class SlicingOpLambda(tf.keras.layers.Layer):
    def __init__(self, function=None, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def call(self, inputs):
        if self.function == 'operators.getitem':
            return inputs[..., :]
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({'function': self.function})
        return config

# --- Function to Load Model from Google Drive ---
@st.cache_resource
def load_model_from_gdrive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download model.")
    model_io = BytesIO(response.content)
    return load_model(model_io, custom_objects={'SlicingOpLambda': SlicingOpLambda})

# --- Page Title ---
st.title("EEG Harmful Brain Activity Classifier")

# --- Google Drive File ID (Edit this) ---
file_id = "YOUR_FILE_ID_HERE"  # Replace with actual file ID

# --- Load Model ---
try:
    model = load_model_from_gdrive(file_id)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")

# --- Upload EEG Input Data ---
uploaded_file = st.file_uploader("Upload EEG CSV", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())

        # Preprocess (you can adjust this depending on your model's expected input)
        input_array = data.values.astype(np.float32)
        input_array = np.expand_dims(input_array, axis=0)

        prediction = model.predict(input_array)
        st.write("Prediction Output:", prediction)

        if prediction[0][0] > 0.5:
            st.warning("⚠️ Potential harmful brain activity detected.")
        else:
            st.success("✅ No harmful activity detected.")
    except Exception as e:
        st.error(f"Error in processing file: {str(e)}")
