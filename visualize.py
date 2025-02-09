"""
BraTS Dataset Evaluation Script

This script loads a trained Unet model and evaluates it on the BraTS dataset,
providing visualization capabilities for the results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython.display import display
from collections import OrderedDict
from torchvision import transforms
from tqdm import tqdm

from model.dataloader.bratsloader import BRATSDataset3D
from model.unet import UNet


def setup_paths():
    """Define and return paths used in the script."""
    return {
        'brats_dir': 'C:/Users/admin/Desktop/Spring2025/AIPI540/CV/MICCAI_BraTS2020_TrainingData',
        'val_dir': 'uploaded_files',
        'output_dir': 'C:/Users/admin/Desktop/Spring2025/AIPI540/CV/A4-Unet/evaluate_result',
        'checkpoint_path': 'checkpoints/sspp_checkpoint_epoch5.pth'
    }


def initialize_model(checkpoint_path, device):
    """
    Initialize the A4-Unet model and load checkpoints.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on
        
    Returns:
        torch.nn.Module: Initialized model
    """
    model = UNet(
        n_channels=4,
        n_classes=2,
        bilinear=False,
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_state_dict = OrderedDict(
        (k, v) for k, v in state_dict.items() if k != 'mask_values'
    )
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model


def prepare_dataset(val_dir):
    """
    Prepare the BraTS dataset with appropriate transformations.
    
    Args:
        val_dir (str): Directory containing validation data
        
    Returns:
        BRATSDataset3D: Prepared dataset
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True)
    ])
    return BRATSDataset3D(val_dir, transform=transform, test_flag=False)


def predict_masks(model, dataloader, device):
    """
    Generate predictions for all images in the dataloader.
    
    Args:
        model (torch.nn.Module): The trained model
        dataloader (BRATSDataset3D): Dataset loader
        device (torch.device): Device to perform computations on
        
    Returns:
        tuple: Lists containing images, true masks, and predicted masks
    """
    images = []
    true_masks = []
    pred_masks = []
    
    for slice_data in tqdm(dataloader, desc='Predict masks', unit='slice', leave=False):
        image, mask, _ = slice_data
        image = image.unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
            memory_format=torch.channels_last
        )
        
        with torch.no_grad():
            pred_mask = model(image)
        
        image = image.squeeze(0).cpu().numpy()
        pred_mask = pred_mask.argmax(dim=1).squeeze().cpu().numpy()
        
        images.append(image)
        true_masks.append(mask)
        pred_masks.append(pred_mask)
    
    return images, true_masks, pred_masks


def plot_image_and_mask_with_slider(images, masks_true, masks_pred=None, figsize=(12, 6)):
    """
    Create an interactive plot with a slider to view different slices.
    
    Args:
        images (list): List of image arrays
        masks_true (list): List of true mask arrays
        masks_pred (list, optional): List of predicted mask arrays
        figsize (tuple): Figure size (width, height)
    """
    def plot_slice(slice_idx):
        plt.figure(figsize=figsize)
        modality = ['T1', 'T1ce', 'T2', 'FLAIR']
        
        # Plot modalities
        for i, mod in enumerate(modality):
            plt.subplot(2, 4, i + 1)
            plt.imshow(np.squeeze(images[slice_idx][i]), cmap='gray')
            plt.title(mod)
            plt.axis('off')
        
        # Plot masks
        plt.subplot(2, 4, 6)
        plt.imshow(np.squeeze(masks_true[slice_idx]), cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        if masks_pred is not None:
            plt.subplot(2, 4, 7)
            plt.imshow(np.squeeze(masks_pred[slice_idx]), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')
        
        plt.show()
    
    slice_slider = widgets.IntSlider(
        min=0,
        max=len(images) - 1,
        step=1,
        description='Slice'
    )
    interactive_plot = widgets.interactive(plot_slice, slice_idx=slice_slider)
    display(interactive_plot)

def get_visualize_data():
    """
    Get the data for the visualize function.
    """
    # Setup
    paths = setup_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Initialize model and dataset
    model = initialize_model(paths['checkpoint_path'], device)
    dataset = prepare_dataset(paths['val_dir'])

    return predict_masks(model, dataset, device)

def main():
    """Main execution function."""
    # Setup
    paths = setup_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and dataset
    model = initialize_model(paths['checkpoint_path'], device)
    dataset = prepare_dataset(paths['val_dir'])
    
    # Generate predictions
    images, true_masks, pred_masks = predict_masks(model, dataset, device)
    
    # Display results
    plot_image_and_mask_with_slider(images, true_masks, pred_masks)

if __name__ == '__main__':
    main()