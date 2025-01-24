import os
import nibabel as nib

# Define the base directory
base_dir = 'Task01_BrainTumour'

# Define the subdirectories
subdirs = ['imagesTr', 'imagesTs', 'labelsTr']

# Function to load NIfTI files from a directory
def load_nifti_files(directory):
    nifti_data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.nii'):
            file_path = os.path.join(directory, filename)
            print(f"Loading {file_path}...")  # Add logging
            try:
                nifti_img = nib.load(file_path)
                nifti_data[filename] = nifti_img.get_fdata()
                print(f"Loading {file_path}...")  # Add logging
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return nifti_data

# Load data from each subdirectory
data = {}
for subdir in subdirs:
    dir_path = os.path.join(base_dir, subdir)
    data[subdir] = load_nifti_files(dir_path)

# Example: Accessing data
print(f"Loaded {len(data['imagesTr'])} training images.")
print(f"Loaded {len(data['imagesTs'])} test images.")
print(f"Loaded {len(data['labelsTr'])} training labels.")



"""
import numpy as np
import torch
from PIL import Image

from os.path import splitext, isfile, join


def load_image(filename):

    ext = splitext(filename)[1]

    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    
    else:
        return Image.open(filename)
    

        



def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob)



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir)
    
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale, medsegdiff, input_size):
        super().__init__(images_dir, mask_dir, scale, medsegdiff, input_size, mask_suffix='_mask')

"""
