# MNIST-Autoencoder-RProp-Analysis

This repository explores the impact of dimensionality reduction on handwritten digit classification (MNIST) by comparing multiple **Autoencoder** architectures and utilizing the **Resilient Backpropagation (RProp)** algorithm for weight updates.

## 📌 Project Overview
The core objective is to analyze how the size of the latent representation ($m_h$) affects learning dynamics and final classification accuracy. The project implements $k=5$ independent Autoencoders, each with a single hidden layer of size $m_h \in \{25, 50, 75, 100, 125\}$.

The latent features extracted by these encoders ($E_h$) serve as inputs for a neural classifier with sigmoid activation functions, trained using **Batch RProp**.

## 🛠️ Technical Specifications
*   **Dataset:** MNIST (N input–label pairs, split into ≥10,000 training and ≥2,500 test samples).
*   **Feature Extraction:** $k$ single-hidden-layer Autoencoders.
*   **Classifier:** Single-hidden-layer Neural Network with Sigmoid activations.
*   **Optimization:** Resilient Backpropagation (RProp) in strict **Batch Mode**.
*   **Metrics:** Evaluation of convergence speed (epochs) and top-1 accuracy on the test set.

## 📊 Learning Dynamics Analysis
The project evaluates performance across five different latent space dimensions:

| Model | Latent Size ($m_h$) | Epochs to Converge | Reconstruction Error| Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Encoder 1 | 25 | 100 | 0.0351417 | 88.21% |
| Encoder 2 | 50 | 100 | 0.0276789 | 93.32% |
| Encoder 3 | 75 | 100 | 0.0203881 | 95.69% |
| Encoder 4 | 100 | 100 | 0.0163986 | 95.31% |
| Encoder 5 | 125 | 100 | 0.0146692 | 96.34% |

## 🚀 Compilation and Setup
To compile the project, navigate to the `src` directory containing the `Makefile` and run the `make` command:

```bash
cd src
make