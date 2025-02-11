# Converted from Jupyter Notebook

# Import necessary libraries
import numpy as np
import pandas as pd
import tarfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import init_notebook_mode, plot, iplot

import nibabel as nib
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, log_loss

# Initialize Plotly
init_notebook_mode(connected=True)

# Extract dataset
def extract_task1_files(root="./data"):
    tar = tarfile.open("data/BraTS2021_Training_Data.tar")
    tar.extractall("data/traning_data")
    tar.close()

extract_task1_files()

# Image Reader Class
class ImageReader:
    def __init__(self, root: str, img_size: int = 256, normalize: bool = False, single_class: bool = False) -> None:
        pad_size = 256 if img_size > 256 else 224
        self.resize = A.Compose([
            A.PadIfNeeded(min_height=pad_size, min_width=pad_size, value=0),
            A.Resize(img_size, img_size)
        ])
        self.normalize = normalize
        self.single_class = single_class
        self.root = root

    def read_file(self, path: str) -> dict:
        scan_type = path.split('_')[-1]
        raw_image = nib.load(path).get_fdata()
        raw_mask = nib.load(path.replace(scan_type, 'seg.nii.gz')).get_fdata()
        processed_frames, processed_masks = [], []
        
        for frame_idx in range(raw_image.shape[2]):
            frame = raw_image[:, :, frame_idx]
            mask = raw_mask[:, :, frame_idx]
            resized = self.resize(image=frame, mask=mask)
            processed_frames.append(resized['image'])
            processed_masks.append(1 * (resized['mask'] > 0) if self.single_class else resized['mask'])
        
        scan_data = np.stack(processed_frames, 0)
        if self.normalize and scan_data.max() > 0:
            scan_data = (scan_data / scan_data.max()).astype(np.float32)
        
        return {
            'scan': scan_data,
            'segmentation': np.stack(processed_masks, 0),
            'orig_shape': raw_image.shape
        }

    def load_patient_scan(self, idx: int, scan_type: str = 'flair') -> dict:
        patient_id = str(idx).zfill(5)
        scan_filename = f'{self.root}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_{scan_type}.nii.gz'
        return self.read_file(scan_filename)

# 3D Image Visualization Class
class ImageViewer3d:
    def __init__(self, reader: ImageReader, mri_downsample: int = 10, mri_colorscale: str = 'Ice') -> None:
        self.reader = reader
        self.mri_downsample = mri_downsample
        self.mri_colorscale = mri_colorscale

    def load_clean_mri(self, image: np.array, orig_dim: int) -> dict:
        shape_offset = image.shape[1] / orig_dim
        z, x, y = (image > 0).nonzero()
        x, y, z = x[::self.mri_downsample], y[::self.mri_downsample], z[::self.mri_downsample]
        colors = image[z, x, y]
        return dict(x=x / shape_offset, y=y / shape_offset, z=z, colors=colors)

    def load_tumor_segmentation(self, image: np.array, orig_dim: int) -> dict:
        tumors = {}
        shape_offset = image.shape[1] / orig_dim
        sampling = {1: 1, 2: 3, 4: 5}
        for class_idx in sampling:
            z, x, y = (image == class_idx).nonzero()
            x, y, z = x[::sampling[class_idx]], y[::sampling[class_idx]], z[::sampling[class_idx]]
            tumors[class_idx] = dict(x=x / shape_offset, y=y / shape_offset, z=z, colors=class_idx / 4)
        return tumors

    def get_3d_scan(self, patient_idx: int, scan_type: str = 'flair') -> go.Figure:
        scan = self.reader.load_patient_scan(patient_idx, scan_type)
        clean_mri = self.load_clean_mri(scan['scan'], scan['orig_shape'][0])
        tumors = self.load_tumor_segmentation(scan['segmentation'], scan['orig_shape'][0])
        
        data = [
            go.Scatter3d(x=clean_mri['x'], y=clean_mri['y'], z=clean_mri['z'],
                         mode='markers', marker=dict(size=3, opacity=0.3, color=clean_mri['colors'], colorscale=self.mri_colorscale)),
            go.Scatter3d(x=tumors[1]['x'], y=tumors[1]['y'], z=tumors[1]['z'],
                         mode='markers', marker=dict(size=3, opacity=0.8, color=tumors[1]['colors'])),
            go.Scatter3d(x=tumors[2]['x'], y=tumors[2]['y'], z=tumors[2]['z'],
                         mode='markers', marker=dict(size=3, opacity=0.4, color=tumors[2]['colors'])),
            go.Scatter3d(x=tumors[4]['x'], y=tumors[4]['y'], z=tumors[4]['z'],
                         mode='markers', marker=dict(size=3, opacity=0.4, color=tumors[4]['colors']))
        ]
        return go.Figure(data=data)

# Load and process data
reader = ImageReader('data/traning_data', img_size=128, normalize=True, single_class=False)
viewer = ImageViewer3d(reader, mri_downsample=25)

# Data Preprocessing and Model Training
df = pd.read_csv('data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
targets = dict(zip(df.BraTS21ID, df.MGMT_value))

# Feature Extraction
features = []
for patient_idx in targets:
    try:
        data = reader.load_patient_scan(patient_idx)
        scan_px = (data['scan'] > 0).sum()
        tumor_px = (data['segmentation'] > 0).sum()
        core_px = (data['segmentation'] == 4).sum()
        patient_features = [patient_idx, targets[patient_idx], scan_px, tumor_px, core_px]
        features.append(patient_features)
    except FileNotFoundError:
        continue

df_features = pd.DataFrame(features, columns=['idx', 'target', 'scan_px', 'tumor_px', 'core_px']).set_index('idx')

# Model Training
X, y = df_features.drop('target', axis=1).values, df_features['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

models = [RidgeClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=60)]
ensemble_predictions = [model.fit(X_train, y_train).predict(X_test) for model in models]
ensemble_score = roc_auc_score(y_test, np.mean(ensemble_predictions, 0))

print(f'Ensemble Model ROC AUC Score: {ensemble_score}')
