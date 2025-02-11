# Brain Tumor Segmentation

This project aims to tackle the challenge of brain tumor segmentation using various machine learning approaches, including a naive mean model, classical ML techniques, and a NN-based deep learning model (U-Net + SSPP).

## Problem Statement

Brain tumor segmentation remains challenging due to high variability in MRI images, unclear boundaries, and irregular tumor shapes and textures. This project compares different approaches to address these challenges.

## Data Sources

The BraTS datasets are part of the Brain Tumor Segmentation Challenge. We select the BraTS 2020 datasets as the experimental data for our project. These are publicly available via the following links [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html). All BraTS multimodal scans are provided as NIfTI files (.nii.gz) and include the following: 

- native T1-weighted scans (T1N), 

- post-contrast T1-weighted scans (T1C/T1CE, also referred to as T1Gd)

- T2-weighted scans (T2W/T2)

- T2 Fluid Attenuated Inversion Recovery scans (T2F/FLAIR). 

The training and validation sets have unspecified glioma classifications, and all data underwent standardized preprocessing by the challenge organizers.

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

A U-Net architecture with Swin Spatial Pyramid Pooling (SSPP) was used to incorporate multi-scale information. We tackle this by modifying the bottleneck layer to include Swin Spatial Pyramid Pooling (SSPP) and a Cross-Contextual Attention module shown in ![Figure](https://github.com/WendyWAAAAANG/Medical_Image_Segmentation_APP/blob/main/Deep_Learning_Model/img/SSPP.png)

In DeepLab V3+, [Chen et al.](https://arxiv.org/abs/1706.05587) introduced the Atrous Spatial Pyramid Pooling (ASPP) module, which dynamically selects convolutional blocks of varying sizes to handle different target scales. This approach prevents large targets from being fragmented and maintains long-distance dependencies without altering the network structure.

Inspired by SSPP by [Azad et al.](https://arxiv.org/abs/2208.00713), we replace four dilated convolutions with Swin Transformers to better capture long-range dependencies. The extracted features are merged and fed into a cross-contextual attention module. This enhances the model's ability to capture contextual dependencies across different scales.

The ASPP concatenates feature maps via depth-wise separable convolution, which does not capture channel dependencies. To address this, Azad introduced cross-contextual attention after SSPP feature fusion. Assume each SSPP layer has tokens ($P$) and embedding dimension ($C$) as ($z_{m}^{P \times C}$), representing objects at different scales. We create a multi-scale representation $z_{all}^{P \times MC} = [z_1||z_2...||z_M]$ by concatenating these features. A scale attention module then emphasizes each feature map's contribution, using global representation and an MLP layer to generate scaling coefficients ($w_{scale}$), enhancing contextual dependencies:

$$
w_{scale} = \sigma(W_2\delta(W_1GAP_{z_{all}})),
$$

$$
z_{all}' = w_{scale} \cdot z_{all},
$$


* where $W_1$ and $W_2$ are learnable MLP parameters, $\delta$ is the ReLU function, $\sigma$ is the Sigmoid function, and GAP is global average pooling.

In the second attention level, Cross-Contextual Attention learns scaling parameters to enhance informative tokens by calculating their weight maps, using the same strategy:

$$
w_{tokens} = \sigma(W_3\delta(W_4GAP_{z_{all}'})),
$$

$$
z_{all}'' = w_{tokens} \cdot z_{all}',
$$


## Model Evaluation Process & Metrics

### Metrics
The primary evaluation metric used was the Dice Similarity Coefficient (DSC).

The **Dice Similarity Coefficient (DSC)** is a key metric for evaluating segmentation models, ranging from **0 to 1**, representing the similarity between two samples. It is calculated as:

$$
DSC = \frac{2TP}{FN+FP+2TP}
$$

where:

- **TP** represents **true positive** pixels.
- **FP** indicates **false positive** pixels.
- **FN** represents **false negative** pixels.

### Implementation Details
All experiments are implemented in PyTorch 2.6 and trained on a single GeForce GTX 4090 GPU with 24 GB memory. We use standard back-propagation with the AdamW optimizer and Softmax activation function. Training employs a batch size of 16, an initial learning rate of 1e-5, and runs for 20 epochs. Total training time varies by dataset size: approximately 20 hours for BraTS 2020.


### Experiment Results
The models achieved the following scores:

| Model                     | Dice Score |
|---------------------------|------------|
| Naive Model               | 56.39      |
| Classical ML (Ensemble)   | 58.21      | 
| U-Net (Baseline)          | 64.79      |
| U-Net + SSPP (Ours)       | 85.37 (↑31.76%)       |

## Demo & Conclusion

A live demo of the brain tumor segmentation application is available [here](https://ztckcwal37wc8lugazm2ey.streamlit.app/).

A video demo is available [here](https://www.youtube.com/watch?v=kiEGwjW3Hzk).

The U-Net + SSPP model outperformed the naive and classical ML approaches, achieving a Dice score of 85.3. Further improvements could be made by exploring additional architectures and incorporating more training data.

## Ethics Statement

- Patient privacy and data security: Fully anonymized datasets were used, and the project complies with HIPAA & GDPR regulations.
- Transparency and reproducibility: The code is publicly available, and the methodology is detailed for replication.
- Bias and fairness: The models were evaluated on multiple datasets, and efforts to mitigate biases are ongoing.
- Clinical reliability and safety: The models are not intended for direct clinical use and require further validation.
- Responsible deployment: Future work includes improving interpretability and reducing biases.

## Repository Structure

```sh
.
├── Deep_Learning_Model
│   ├── LICENSE
│   ├── README.md
│   ├── img/
│   ├── model/
│   ├── sample_data/
│   ├── train.py
│   └── visualize.py
├── Machine_Learning_Model
│   ├── LICENSE
│   ├── Machine_Learning_Model.py
│   ├── README.md
│   ├── read_data.sh
│   └── requirements.txt
├── Naive_Model
│   ├── LICENSE
│   ├── README.md
│   ├── naive-mean-model.py
│   ├── output/
│   └── requirements.txt
├── README.md
├── requirements.txt
├── sspp_unet_api
│   ├── Dockerfile
│   ├── README.md
│   ├── app/
│   └── requirements.txt
└── streamlit_app/
```

## Setup and Run Instructions

Tested on `python=3.11`

Install dependencies:
```sh
pip install -r requirements.txt
```

Directory `/sspp_unet_api` contains all the code for our FastAPI backend, which provides four HTTP endpoint methods.

- `/`: Endpoint to health check, print `OK`.
- `/predict`: Endpoint to predict the masks for the uploaded brain tumor slices.
- `/upload`: Endpoint to upload a ZIP file containing brain tumor slices. Extracts files and stores them in a designated directory.
- `/download_sample`: Endpoint to provide a sample dataset for users to download.

To test the FastAPI locally:
```sh
cd sspp_unet_api/app/
python main.py
```

To create a docker image and test locally:
```sh
docker build -t <image_name> .
docker run -d <image_name>
```

If successful, you should see your FastAPI application running on `localhost:8000`. You can check `localhost:8000/docs` to test each endpoint method.

--- 

Directory `/streamlit_app` contains Streamlit frontend allows users to: 
1. Upload files for predictions 
2. Download sample brain tumor data 
3. Visualize brain slices, true masks, and predicted masks.

To test the Streamlit app locally:
```sh
cd streamlit_app/
streamlit run app.py
```
