# Automated End-to-End Machine Vision System for Galaxy Morphology Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-CV2-green)
![Accuracy](https://img.shields.io/badge/Accuracy-82%25-success)
![MAE](https://img.shields.io/badge/Validation_MAE-0.1086-lightgrey)

## 🌌 Overview & Problem Statement
[cite_start]The field of extragalactic astronomy has firmly entered the domain of "Big Data"[cite: 92]. [cite_start]Legacy projects like the Sloan Digital Sky Survey (SDSS) and upcoming initiatives like the Rubin Observatory capture millions to billions of celestial images, making manual human classification practically impossible[cite: 92, 149]. [cite_start]While crowdsourced citizen science initiatives like "Galaxy Zoo" have provided valuable labeled datasets, they are too time-consuming for live survey classification[cite: 94, 153].

[cite_start]This project implements an automated, end-to-end Machine Vision and Deep Learning pipeline to classify galaxy morphologies based on the Hubble Tuning Fork sequence[cite: 97, 117]. 

### The Computer Vision Challenges
Automating this classification presents unique hurdles not found in standard ImageNet tasks:
1. [cite_start]**Rotational Invariance:** In space, there is no "up" or "down"[cite: 158]. [cite_start]A standard Convolutional Neural Network (CNN) views a spiral galaxy rotated at 90° and 45° as two entirely different pixel grids[cite: 96, 161]. 
2. [cite_start]**Low Signal-to-Noise Ratio (SNR):** Astronomical images consist predominantly of empty black space, background noise, cosmic rays, and foreground stars, which easily confuse standard feature extraction algorithms[cite: 164, 166].

## 🎯 Project Objectives
1. [cite_start]**Machine Vision Preprocessing:** Develop an OpenCV pipeline to automatically crop, center, and isolate galaxies from background space[cite: 174].
2. [cite_start]**Architecture:** Design a rotation-invariant Deep Learning model utilizing spatial augmentation layers to classify objects regardless of orientation[cite: 175].
3. [cite_start]**Evaluation:** Benchmark the system against the debiased "Hart16" dataset across 5 probabilistic morphological categories[cite: 176].

---

## 🔬 Target Morphological Classes
[cite_start]The model evaluates images as a multi-label regression task, outputting simultaneous probabilities for five distinct classes[cite: 118, 359]:
* [cite_start]**Smooth:** Featureless ellipsoids resembling a glowing ball of light[cite: 119, 120].
* [cite_start]**Edge-On:** Disk galaxies viewed directly from the side, appearing as a thin, flat line[cite: 121, 122].
* [cite_start]**Spiral:** Complex rotating disks with winding arms and high spatial frequency[cite: 124, 125].
* [cite_start]**Barred:** Galaxies containing a distinct central bar-shaped structure of stars extending from the core[cite: 126].
* [cite_start]**Irregular:** Chaotic, asymmetrical structures often resulting from gravitational collisions[cite: 128, 129].

---

## ⚙️ System Architecture & Methodology

![High-Level System Architecture](docs/architecture_diagram.png)
*(Note: Upload Figure 3 from your report here)*

### 1. Classical Machine Vision Preprocessing (OpenCV)
[cite_start]Processing raw astronomical images filled with 80% blank space is highly inefficient[cite: 262, 263]. [cite_start]A classical computer vision pipeline was engineered to isolate the Region of Interest (ROI)[cite: 264]:
1. [cite_start]**Grayscale Conversion:** Dimensionality reduction to focus on structural intensity[cite: 266].
2. [cite_start]**Gaussian Blurring:** A $5\times5$ kernel was applied to smooth high-frequency cosmic ray artifacts while preserving low-frequency structural details[cite: 267, 268].
3. [cite_start]**Fixed Binary Thresholding:** An empirically determined threshold of 25 was used to separate faint foreground signals (spiral arms) from the dark background[cite: 269, 270].
4. [cite_start]**Morphological Dilation:** A $3\times3$ kernel (2 iterations) reconnected "broken" or disjointed pixels of faint spiral arms to ensure a contiguous contour[cite: 272, 274].
5. [cite_start]**Contour Detection & ROI Cropping:** The algorithm isolates the largest contour, computes a bounding box, and crops the raw image, delivering a high-signal input to the neural network[cite: 275, 277].

### 2. Rotation-Invariant CNN Architecture
[cite_start]To solve the spatial invariance problem without massive dataset disk-bloat, a custom CNN was built[cite: 314, 318]:
* [cite_start]**Spatial Invariance Input Block:** Utilizes dynamic Keras Preprocessing Layers during training, including `RandomRotation(0.5)` for full $\pm180^{\circ}$ coverage, `RandomFlip`, and `RandomZoom(0.1)`[cite: 319, 320, 321].
* [cite_start]**Feature Extraction Base:** Four progressive convolutional blocks (32, 64, 128, 256 filters)[cite: 326]. [cite_start]Each block incorporates `Conv2D`, `Batch Normalization` for accelerated convergence, `LeakyReLU` ($\alpha=0.1$) to prevent dying gradients, and `MaxPooling`[cite: 328, 329, 330, 331].
* [cite_start]**Classification Head:** Replaces standard Flattening with `Global Average Pooling` to heavily reduce parameters, feeding into a 5-unit `Sigmoid` output layer[cite: 332, 334].

---

## 📊 Dataset & Training Configurations
* [cite_start]**Data Fusion:** Fused SDSS raw JPEG imagery with Hart16 morphological labels using Pandas, keyed via `dr7objid`[cite: 247, 249, 253].
* [cite_start]**Sampling:** A stratified random sample of 100,000 images, heavily filtered to remove pure artifact/star images (>50% artifact probability)[cite: 256, 259].
* [cite_start]**Loss Function:** Mean Squared Error (MSE) was utilized to treat the classification as a multi-label regression/probability task[cite: 356, 359].
* [cite_start]**Optimizer:** Adam ($\eta=0.001$)[cite: 361].
* [cite_start]**Regularization:** `ReduceLROnPlateau` dynamically reduced the learning rate, paired with `EarlyStopping` and a 40% `Dropout` rate[cite: 368, 369, 372, 374].

---

## 🏆 Key Results & Performance

[cite_start]The model successfully generalized across 35 epochs, effectively mitigating overfitting and demonstrating exceptional performance on subjective, ambiguous data[cite: 379, 430].

| Metric | Training Score | Validation Score |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | 0.1079 | [cite_start]**0.1086** [cite: 382] |
| **Mean Squared Error (MSE)** | 0.0257 | [cite_start]**0.0258** [cite: 382] |
| **Overall Accuracy** | - | [cite_start]**82%** [cite: 382] |

[cite_start]The validation MAE of ~0.11 signifies that the model's predicted morphological probability deviates by no more than 10.8% from the human consensus crowd-vote[cite: 383, 384].

### ROC Analysis (Area Under Curve)
* [cite_start]**Smooth:** 0.98 [cite: 477]
* [cite_start]**Edge-On:** 0.97 [cite: 477]
* [cite_start]**Irregular:** 0.95 [cite: 477]
* [cite_start]**Spiral:** 0.92 [cite: 477]
* [cite_start]**Barred:** 0.90 [cite: 477]

*(Note: Add Figure 8 - ROC Curves here)*

---

## 💻 Setup and Installation
*Note: Due to the extreme file sizes of raw astronomical imagery, the 100k SDSS images are not hosted in this repository.*

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/galaxy-morphology-classification.git](https://github.com/your-username/galaxy-morphology-classification.git)
   cd galaxy-morphology-classification
