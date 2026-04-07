# Autoencoder Marine Data Reconstructor

This repository contains the third practical assignment (**TP3**) for the **INF7370 - Machine Learning** course at UQAM. The project focuses on unsupervised learning using a **Convolutional Autoencoder (CAE)** to learn compact latent representations (embeddings) of marine animal images.

## Project Overview
The system is designed to compress RGB images of dolphins and sharks into a reduced feature space and then reconstruct them as accurately as possible.

* **Task:** Image reconstruction and feature extraction (unsupervised).
* **Dataset:** 4,200 total images (3,600 training/validation, 600 test).
* **Target Classes:** Dolphins and Sharks.
* **Input Dimensions:** 140x140 pixels, 3 channels (RGB).

## Architecture Design
The model utilizes a symmetrical encoder-decoder structure with 335,747 total parameters.

### 1. Encoder (Feature Extraction)
* **Layers:** Three convolutional blocks.
* **Components:** `Conv2D` layers (32, 64, 128 filters), `BatchNormalization`, `ReLU` activation, and `MaxPooling2D`.
* **Regularization:** `Dropout` (0.2) to prevent overfitting.
* **Output:** Generates a compact **Embedding** vector used for downstream tasks.

### 2. Decoder (Reconstruction)
* **Layers:** Three symmetrical upsampling blocks.
* **Components:** `Conv2D` layers followed by `UpSampling2D` to restore spatial dimensions.
* **Final Layer:** `Conv2D` with **Sigmoid** activation to ensure output pixels are normalized between [0, 1].
* **Post-processing:** `Cropping2D` to precisely match the 140x140 input size.

## Training & Hyperparameters
* **Loss Function:** Mean Squared Error (MSE).
* **Optimizer:** Adam (Initial Learning Rate: 0.001).
* **Batch Size:** 16.
* **Data Processing:** Global normalization (rescale 1/255). No data augmentation was used to ensure reconstruction fidelity.
* **Callbacks:**
    * **EarlyStopping:** Stopped training at epoch 73 to prevent overfitting.
    * **ReduceLROnPlateau:** Halved learning rate after 5 epochs of stagnation.

## 📊 Evaluation & Results
The model achieved a stable convergence with a final validation loss of **0.003324**.

### 1. Visual Reconstruction
The CAE effectively preserves general shapes and silhouettes of dolphins and sharks, though fine details remain slightly blurred due to the compression bottleneck.

### 2. SVM Classification (Latent Space Quality)
To validate the informativeness of the learned embeddings, a Linear SVM was trained to classify species:
* **Accuracy (Raw Pixels):** 59.00%.
* **Accuracy (Learned Embeddings):** **68.00%**.
* **Conclusion:** The encoder successfully extracted discriminative features without explicit labels, providing a 9% performance gain over raw data.

### 3. t-SNE Visualization
A 2D projection of the embeddings via t-SNE shows a reasonable separation between the dolphin and shark classes, confirming the structural integrity of the latent space.

## Repository Structure
```text
.
├── 1_Modele_TP3.py      # Training script and CAE architecture
├── 2_Evaluation_TP3.py  # SVM evaluation, t-SNE, and reconstruction plots
├── README.md            # Project documentation
└── INF7370_H25_TP3...   # Detailed technical report (PDF)
```

**Authors**: Francois Gonothi Toure & Martial Zachee Kaljob Kollo
