import numpy as np
import nibabel as nib
from pathlib import Path
import json
import matplotlib.pyplot as plt

class NaiveMeanModel:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.mean_mask = None
        # Load dataset info
        with open(self.base_path / 'dataset.json', 'r') as f:
            self.dataset_info = json.load(f)
    
    def load_case(self, case_id):
        """Load a single case with all modalities"""
        case_data = next(
            (case for case in self.dataset_info['training'] 
             if case['image'].split('/')[-1].startswith(case_id)),
            None
        )
        
        if case_data is None:
            raise ValueError(f"Case {case_id} not found in dataset")
            
        label_path = self.base_path / case_data['label'].lstrip('./')
        label = nib.load(str(label_path))
        label_data = label.get_fdata()
        
        return label_data
    
    def train(self, training_cases, num_cases=None):
        """
        Create mean tumor mask from training cases
        """
        # Use all cases if num_cases is None
        if num_cases is None:
            num_cases = len(training_cases)
            
        print(f"Training on {num_cases} cases...")
        all_masks = []
        class_counts = {0: 0, 1: 0}
        
        # Load and accumulate masks
        for case_id in training_cases[:num_cases]:
            print(f"Processing case {case_id}")
            mask = self.load_case(case_id)
            all_masks.append(mask)
            
            # Count actual presence of each class
            for label in [0, 1]:
                if np.any(mask == label):
                    class_counts[label] += 1

        # Calculate mean mask
        self.mean_mask = np.mean(np.stack(all_masks, axis=0), axis=0)
        
        # Calculate true class frequencies
        class_freqs = {}
        total_cases = len(all_masks)
        for label in [0, 1]:
            freq = class_counts[label] / total_cases
            class_freqs[f'class_{label}_freq'] = freq
        
        print("\nClass frequencies in training data:")
        for label, freq in class_freqs.items():
            print(f"{label}: {freq:.4f}")
        
        return self.mean_mask, class_freqs
    
    def predict(self, shape):
        """
        Predict using the mean mask
        """
        if self.mean_mask is None:
            raise ValueError("Model must be trained before prediction")
            
        # Convert probabilities to labels based on relative probabilities
        prediction = np.zeros_like(self.mean_mask)
        
        # Get probability ranges for each class
        p_min = np.min(self.mean_mask)
        p_max = np.max(self.mean_mask)
        p_range = p_max - p_min
        
        # Set adaptive thresholds based on probability distribution
        thresholds = {
            0: p_min + p_range * 0.3,  # Lower threshold for edema
            1: p_min + p_range * 0.5,  # Middle threshold for non-enhancing
        }
        
        # Apply thresholds in reverse order (highest to lowest)
        for label in [0, 1]:
            prediction[self.mean_mask >= thresholds[label]] = label
        
        return prediction
    
    def evaluate(self, true_mask, pred_mask):
        """
        Evaluate segmentation performance
        """
        scores = {}
        for label in [0, 1]:
        # True Positives, False Positives, False Negatives
            true_label = (true_mask == label)
            pred_label = (pred_mask == label)

            TP = np.sum(true_label & pred_label)
            FP = np.sum(~true_label & pred_label)
            FN = np.sum(true_label & ~pred_label)
            
            # Dice Similarity Coefficient
            dice = (2 * TP) / (FN + FP + 2 * TP + 1e-8)
            scores[f'dice_label_{label}'] = dice
        
        scores['mean_dice'] = np.mean(list(scores.values()))
        return scores

def visualize_results(true_mask, pred_mask, mean_mask, slice_idx, save_path=None):
    """
    Visualize results including mean probability map
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Mean Probability Map')
    plt.imshow(mean_mask[:, :, slice_idx], cmap='jet')
    plt.colorbar()
    
    plt.subplot(132)
    plt.title('True Mask')
    plt.imshow(true_mask[:, :, slice_idx])
    
    plt.subplot(133)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask[:, :, slice_idx])
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def split_dataset(training_cases, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Split dataset into training, validation, and test sets
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Shuffle the cases
    cases = np.array(training_cases)
    np.random.shuffle(cases)
    
    # Calculate split indices
    n_total = len(cases)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split the data
    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train+n_val]
    test_cases = cases[n_train+n_val:]
    
    # Print split information
    print(f"\nDataset Split:")
    print(f"Total cases: {n_total}")
    print(f"Training: {len(train_cases)} cases")
    print(f"Validation: {len(val_cases)} cases")
    print(f"Test: {len(test_cases)} cases")
    
    return train_cases.tolist(), val_cases.tolist(), test_cases.tolist()

def main():
    # Set paths
    base_path = 'Task01_BrainTumour'
    
    # Initialize model
    model = NaiveMeanModel(base_path)
    
    # Get list of training cases
    with open(Path(base_path) / 'dataset.json', 'r') as f:
        dataset_info = json.load(f)
    training_cases = [Path(case['image']).stem.split('.')[0] for case in dataset_info['training']]
    
    # Split dataset
    train_cases, val_cases, test_cases = split_dataset(training_cases)
    
    # Train model using training set
    print("\nTraining model...")
    mean_mask, class_freqs = model.train(train_cases)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_scores = []
    for case_id in val_cases:
        true_mask = model.load_case(case_id)
        pred_mask = model.predict(true_mask.shape)
        scores = model.evaluate(true_mask, pred_mask)
        val_scores.append(scores)
    
    # Calculate mean validation scores
    mean_val_scores = {}
    for metric in val_scores[0].keys():
        mean_val_scores[metric] = np.mean([score[metric] for score in val_scores])
    
    print("\nValidation Performance:")
    for metric, value in mean_val_scores.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_scores = []
    for case_id in test_cases:
        true_mask = model.load_case(case_id)
        pred_mask = model.predict(true_mask.shape)
        scores = model.evaluate(true_mask, pred_mask)
        test_scores.append(scores)
    
    # Calculate mean test scores
    mean_test_scores = {}
    for metric in test_scores[0].keys():
        mean_test_scores[metric] = np.mean([score[metric] for score in test_scores])
    
    print("\nTest Performance:")
    for metric, value in mean_test_scores.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize example from test set
    test_case = test_cases[0]
    true_mask = model.load_case(test_case)
    pred_mask = model.predict(true_mask.shape)
    
    # Find slice with maximum tumor area
    tumor_areas = [np.sum(true_mask[:,:,i] > 0) for i in range(true_mask.shape[2])]
    best_slice = np.argmax(tumor_areas)
    
    # Visualize
    visualize_results(
        true_mask,
        pred_mask,
        mean_mask,
        slice_idx=best_slice,
        save_path="mean_model_results.png"
    )
    print("\nVisualization saved as 'mean_model_results.png'")

if __name__ == "__main__":
    main()