# Deep Learning Model -- SSPP-Unet

## Overview
A U-Net architecture with Swin Spatial Pyramid Pooling (SSPP) was used to incorporate multi-scale information. We tackle this by modifying the bottleneck layer to include Swin Spatial Pyramid Pooling (SSPP) and a Cross-Contextual Attention module shown in ![Figure](https://github.com/WendyWAAAAANG/Medical_Image_Segmentation_APP/blob/Roxanne/Deep_Learning_Model/img/SSPP.png)


## Architecture of SSPP
In DeepLab V3+, [Chen et al.](https://arxiv.org/abs/1706.05587) introduced the Atrous Spatial Pyramid Pooling (ASPP) module, which dynamically selects convolutional blocks of varying sizes to handle different target scales. This approach prevents large targets from being fragmented and maintains long-distance dependencies without altering the network structure.

Inspired by SSPP by [Azad et al.](https://arxiv.org/abs/2208.00713), we replace four dilated convolutions with Swin Transformers to better capture long-range dependencies. The extracted features are merged and fed into a cross-contextual attention module. This enhances the model's ability to capture contextual dependencies across different scales.

The ASPP concatenates feature maps via depth-wise separable convolution, which does not capture channel dependencies. To address this, Azad introduced cross-contextual attention after SSPP feature fusion. Assume each SSPP layer has tokens ($P$) and embedding dimension ($C$) as ($z_{m}^{P \times C}$), representing objects at different scales. We create a multi-scale representation $z_{all}^{P \times MC} = [z_1||z_2...||z_M]$ by concatenating these features. A scale attention module then emphasizes each feature map's contribution, using global representation and an MLP layer to generate scaling coefficients ($w_{scale}$), enhancing contextual dependencies:

$$
w_{scale} = \sigma(W_2\delta(W_1GAP_{z_{all}})),
$$

$$
z_{all}' = w_{scale} \cdot z_{all},
$$

\noindent where $W_1$ and $W_2$ are learnable MLP parameters, $\delta$ is the ReLU function, $\sigma$ is the Sigmoid function, and GAP is global average pooling.

In the second attention level, Cross-Contextual Attention learns scaling parameters to enhance informative tokens by calculating their weight maps, using the same strategy:

$$
w_{tokens} = \sigma(W_3\delta(W_4GAP_{z_{all}'})),
$$

$$
z_{all}'' = w_{tokens} \cdot z_{all}',
$$


## Experiment Result

The primary evaluation metric used was the Dice Similarity Coefficient (DSC). The models achieved the following scores:

| Model                     | Dice Score |
|---------------------------|------------|
| U-Net (Baseline)          | 64.79      |
| U-Net + SSPP (Ours)       | 85.37 (↑31.76%)       |


## Conclusion
The U-Net + SSPP model outperformed the naive and classical ML approaches, achieving a Dice score of 85.3. Further improvements could be made by exploring additional architectures and incorporating more training data.
