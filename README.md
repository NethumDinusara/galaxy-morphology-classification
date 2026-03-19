# 🌌 Galaxy Morphology Classification: Addressing Rotational Invariance

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-82%25-success)
![MAE](https://img.shields.io/badge/MAE-0.1086-lightgrey)

## 📑 Executive Summary
This repository contains an automated, end-to-end Machine Vision and Deep Learning pipeline designed to classify the morphological structures of galaxies from the Sloan Digital Sky Survey (SDSS). 

The project replicates the probabilistic consensus of human astronomers by categorizing galaxies into five distinct classes: **Smooth, Edge-On, Spiral, Barred, and Irregular**. By combining a classical OpenCV preprocessing pipeline with a custom, rotation-invariant Convolutional Neural Network (CNN), the system successfully processes noisy astronomical imagery and achieves a Mean Absolute Error (MAE) of 0.1086.

---

## 🚀 The Core Challenges Solved

Standard image recognition models struggle with astronomical data due to two primary factors, which this architecture explicitly addresses:

1. **The Rotational Invariance Problem:** In space, there is no "up" or "down." [cite_start]A spiral galaxy viewed at 45° and 130° represents the exact same object, but a standard CNN interprets these as completely different pixel grids.
2. **Low Signal-to-Noise Ratio (SNR):** Raw astronomical images consist of nearly 80% empty black space, filled with cosmic rays and background noise that confuse feature extraction.

---

![Overview](./images/overview.png)

## 🧠 System Architecture

![System Architecture Diagram](./images/architecture.png)

The solution is divided into a robust two-stage pipeline:

### Phase 1: Machine Vision Preprocessing (OpenCV)
Instead of feeding raw, noise-heavy images to the neural network, a 5-step classical computer vision pipeline isolates the Region of Interest (ROI).
* **Gaussian Blurring:** A 5x5 kernel smooths high-frequency "shot noise" while preserving low-frequency galaxy structures.
* **Fixed Binary Thresholding:** Empirically set to a value of 25 to separate the foreground signal from the dark background without discarding faint spiral arms.
* **Morphological Dilation:** A 3x3 kernel (2 iterations) reconnects disjointed pixels of faint spiral arms into a single contiguous contour.
* **Contour Detection & Cropping:** Automatically detects the largest contour, generates a bounding box, and crops the image directly around the galactic core.

### Phase 2: Deep Learning Classification (Custom CNN)
The classification engine is built to dynamically handle the spatial randomness of the cosmos.
* **Spatial Invariance Block:** Utilizes Keras Preprocessing Layers (`RandomRotation` $\pm180^{\circ}$, `RandomFlip`, `RandomZoom`) at the input stage. This forces the network to learn orientation-agnostic features dynamically on the CPU/GPU, eliminating the need for massive static dataset expansion.
* **Hierarchical Feature Extraction:** Four convolutional blocks (32, 64, 128, 256 filters) utilizing `Conv2D`, `Batch Normalization`, `LeakyReLU` ($a=0.1$), and `MaxPooling`.
* **Probabilistic Output:** Utilizes `Global Average Pooling` (to retain spatial data and reduce parameters) feeding into a 5-unit `Sigmoid` output layer. The network is trained using Mean Squared Error (MSE) to predict independent probabilities for each morphological class simultaneously.

---

## 📊 Dataset & Training Strategy

* **Data Fusion:** Fused over 100,000 raw SDSS images from the Galaxy Zoo 2 dataset with "Hart16" debiased volunteer vote fractions.
* **Artifact Filtering:** Rigorous cleaning discarded any image with a $>50\%$ probability of being a foreground star/artifact, ensuring the model trained on pure galactic features.
* **Optimization:** Trained over 35 epochs using the Adam optimizer (initial learning rate $\eta=0.001$), backed by `ReduceLROnPlateau` and `EarlyStopping` callbacks to prevent overfitting. A 40% Dropout layer further ensured strong generalization.

---

## 📈 Key Results & Evaluation

The model achieved highly competitive results, proving its viability for pre-filtering large-scale astronomical data streams. 

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Validation MAE** | `0.1086` | Predictions deviate by <10.8% from human consensus. |
| **Overall Accuracy** | `82%` | Across five highly subjective morphological classes. |
| **Spiral $R^2$ Score** | `0.708` | Strong linear correlation in detecting complex spiral patterns. |

### ROC Analysis & AUC Scores
The model demonstrated exceptional sensitivity and specificity across all classes:
* **Smooth:** 0.98 AUC 
* **Edge-On:** 0.97 AUC 
* **Irregular:** 0.95 AUC 
* **Spiral:** 0.92 AUC 
* **Barred:** 0.90 AUC 

![ROC Curves](./images/roc.png)

### Interpretability
Internal feature map visualizations from `conv2d_4` confirm that the network organically learned to identify galactic cores (bulges) as central blobs, while deeper filters activated on the high-frequency textural changes of spiral arms.

![Final Output](./images/image.png)

---

## 💻 Technologies Used
* **Languages:** Python 3.10 
* **Deep Learning:** TensorFlow / Keras (v2.x) 
* **Computer Vision:** OpenCV (`cv2`) 
* **Data Handling & Analysis:** Pandas, NumPy, Scikit-Learn 
* **Visualization:** Matplotlib, Seaborn

---

## ⚙️ Installation & Usage

1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/galaxy-morphology-classification.git](https://github.com/yourusername/galaxy-morphology-classification.git)
