import streamlit as st
import requests
import base64
import numpy as np
import io
import matplotlib.pyplot as plt

FASTAPI_URL = "https://informally-unbiased-wallaby.ngrok-free.app"

def decode_numpy(encoded_str: str) -> np.ndarray:
    """Decode Base64 string back to NumPy array."""
    buffer = io.BytesIO(base64.b64decode(encoded_str))
    return np.load(buffer, allow_pickle=True)

st.title("Brain Tumor Segmentation Viewer")

# Upload File
st.header("Upload Brain Tumor Slices")
uploaded_file = st.file_uploader("Upload ZIP file", type=["zip"])
if uploaded_file:
    files = {"file": ('BraTS_data.zip', uploaded_file.getvalue())}
    response = requests.post(f"{FASTAPI_URL}/upload", files=files)
    if response.status_code == 200:
        st.success("File uploaded and processed successfully!")
    else:
        st.error("Error uploading file")

# Download Sample Data
st.header("Try with Sample Data")
sample_response = requests.get(f"{FASTAPI_URL}/download_sample")
if sample_response.status_code != 200:
    st.error("Error downloading sample dataset")
st.download_button("Download Sample Data", sample_response.content, "sample_brain_tumor.zip")

# Run Prediction
st.header("Run Model Prediction")
if st.button("Get Predictions"):
    response = requests.post(f"{FASTAPI_URL}/predict")
    if response.status_code == 200:
        data = response.json()
        images = decode_numpy(data["images"])
        true_masks = decode_numpy(data["true_masks"])
        pred_masks = decode_numpy(data["pred_masks"])
        
        st.session_state["images"] = images
        st.session_state["true_masks"] = true_masks
        st.session_state["pred_masks"] = pred_masks
        st.session_state["index"] = 0
    elif response.status_code == 400:
        st.error(response.json()["detail"])
    else:
        st.error("Error fetching predictions")

# Display Prediction Results
if "images" in st.session_state:
    st.header("View Prediction Results")
    index = st.slider("Select Image Slice", 0, len(st.session_state["images"]) - 1, st.session_state["index"])
    st.session_state["index"] = index
    
    modality = ['T1', 'T1ce', 'T2', 'FLAIR']
    fig, axes = plt.subplots(2, 4, figsize=(12, 4))
    for i in range(4):
        axes[0, i].imshow(st.session_state["images"][index][i], cmap="gray")
        axes[0, i].set_title(modality[i])
        axes[0, i].axis("off")
    
    axes[1, 1].imshow(np.squeeze(st.session_state["true_masks"][index]), cmap="gray")
    axes[1, 1].set_title("True Mask")
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(np.squeeze(st.session_state["pred_masks"][index]), cmap="gray")
    axes[1, 2].set_title("Predicted Mask")
    axes[1, 2].axis("off")

    axes[1, 0].axis("off")
    axes[1, 3].axis("off")

    st.pyplot(fig)
