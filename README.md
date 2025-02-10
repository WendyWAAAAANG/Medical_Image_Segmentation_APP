# Brain Tumor Segmentation

This project aims to tackle the challenge of brain tumor segmentation using various machine learning approaches, including a naive mean model, classical ML techniques, and a NN-based deep learning model (U-Net + SSPP).

## Problem Statement

Brain tumor segmentation remains challenging due to high variability in MRI images, unclear boundaries, and irregular tumor shapes and textures. This project compares different approaches to address these challenges.

## Data Sources

The dataset used for this project is the BraTS 2021 Task 1 dataset, which can be found on Kaggle: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data

The dataset includes the following MRI scans:
- Fluid Attenuated Inversion Recovery (FLAIR)
- T1-weighted pre-contrast (T1w)
- T1-weighted post-contrast (T1Gd) 
- T2-weighted (T2)

## Modeling Approaches

### Naive Model: Mean Approach

A simple baseline model using the mean value.

### Classical Machine Learning Models

Several classical ML models were evaluated, including:
- RidgeClassifier
- DecisionTreeClassifier
- KNeighborsClassifier
- Ensemble of All

### Deep Learning Model: U-Net + SSPP

A U-Net architecture with Spatial Pyramid Pooling (SSPP) was used to incorporate multi-scale information. This approach is well-suited for medical images with multi-target and various scales of targets.

## Model Evaluation Process & Metrics

The primary evaluation metric used was the Dice Similarity Coefficient (DSC). The models achieved the following scores:

| Model                     | Dice Score |
|---------------------------|------------|
| Naive Model               | 4.9        |
| Classical ML (best)       | 57.4       | 
| U-Net                     | 56.7       |
| U-Net + SSPP              | 85.3       |

## Demo & Conclusion

A live demo of the brain tumor segmentation application is available [link to demo].

The U-Net + SSPP model outperformed the naive and classical ML approaches, achieving a Dice score of 85.3. Further improvements could be made by exploring additional architectures and incorporating more training data.

## Ethics Statement

- Patient privacy and data security: Fully anonymized datasets were used, and the project complies with HIPAA & GDPR regulations.
- Transparency and reproducibility: The code is publicly available, and the methodology is detailed for replication.
- Bias and fairness: The models were evaluated on multiple datasets, and efforts to mitigate biases are ongoing.
- Clinical reliability and safety: The models are not intended for direct clinical use and require further validation.
- Responsible deployment: Future work includes improving interpretability and reducing biases.

## Repository Structure



## Setup and Run Instructions

