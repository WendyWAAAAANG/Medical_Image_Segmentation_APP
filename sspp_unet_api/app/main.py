from fastapi import FastAPI, UploadFile, File, HTTPException
import subprocess
import shutil
import zipfile
import base64
import io
from pathlib import Path
from fastapi.responses import FileResponse
import numpy as np
from visualize import get_visualize_data
from pydantic import BaseModel

app = FastAPI()

class PredictionResponse(BaseModel):
    images: str
    true_masks: str
    pred_masks: str

UPLOAD_DIR = Path("uploaded_files")
SAMPLE_DATASET_PATH = Path("sample_data/sample_brain_tumor.zip")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

def encode_numpy(array: np.ndarray) -> str:
    """Convert NumPy array to base64 string."""
    with io.BytesIO() as buffer:
        np.save(buffer, array)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.get("/")
async def root():
    return {"health_check": "OK!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict():
    """
    Endpoint to predict the masks for the uploaded brain tumor slices.
    """
    if not UPLOAD_DIR.exists() or not any(UPLOAD_DIR.iterdir()):
        raise HTTPException(status_code=400, detail="No files uploaded for prediction")

    images, true_masks, pred_masks = get_visualize_data()
    
    return {
        "images": encode_numpy(images),
        "true_masks": encode_numpy(true_masks),
        "pred_masks": encode_numpy(pred_masks),
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a ZIP file containing brain tumor slices.
    Extracts files and stores them in a designated directory.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract ZIP file
    extract_dir = UPLOAD_DIR
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    return {"message": "File uploaded and extracted successfully", "extracted_path": str(extract_dir)}


@app.get("/download_sample")
async def download_sample():
    """
    Endpoint to provide a sample dataset for users to download.
    """
    if not SAMPLE_DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    
    return FileResponse(SAMPLE_DATASET_PATH, filename="sample_brain_tumor.zip", media_type="application/zip")

if __name__ == "__main__":
    server_process = subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
    ngrok_process = subprocess.Popen(["ngrok", "http", "--url=informally-unbiased-wallaby.ngrok-free.app", "8000"])

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        server_process.terminate()
        if ngrok_process:
            ngrok_process.terminate()
