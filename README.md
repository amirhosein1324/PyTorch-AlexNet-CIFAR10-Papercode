# AlexNet Implementation for CIFAR-10 Classification

This repository contains a PyTorch implementation of the classic AlexNet Convolutional Neural Network (CNN) trained for image classification on the CIFAR-10 dataset.

---

##  Project Overview

The goal of this project is to implement , train , and evaluate the renowned AlexNet architecture on a smaller, more common benchmark dataset, CIFAR-10, demonstrating the effectiveness of deep convolutional networks for image recognition tasks.

The training process is detailed in the ** AlexNet_CIFAR10.ipynb ** Jupyter Notebook.

---

##  Model Architecture: AlexNet

AlexNet is a deep convolutional network that significantly influenced the field of deep learning , known for its use of stacked convolutional layers , ReLU activation functions , and max-pooling layers.

### Key features of the implementation:

- ** Layer Structure: **  
  The model follows the original AlexNet layout, adapted slightly for the smaller \( 32 \times 32 \) CIFAR-10 images (e.g., adjusting kernel sizes or the final fully connected layers).

- **Training :**  
  Trained using the Adam optimizer and Cross-Entropy Loss over 20 epochs.

- ** Framework: **  
  Built using PyTorch.

---

## Dataset: CIFAR-10

The CIFAR-10 dataset consists of 60,000 \( 32 \times 32 \) color images in 10 classes, with 6,000 images per class.  
The 10 classes are: airplane , automobile , bird , cat , deer , dog , frog , horse , ship, and truck.

The dataset is automatically downloaded and loaded within the notebook using `torchvision.datasets.CIFAR10`.

---

## Results

After 20 epochs of training, the model achieved the following performance on the validation subset of the CIFAR-10 dataset:

| Metric                   | Value   |
|--------------------------|---------|
| Final Validation Accuracy | ** 82.55% ** |

This accuracy demonstrates AlexNets capability to learn robust features even on low-resolution images.

---

## 🛠️ Setup and Requirements

Install the required dependencies:

```bash
pip install torch torchvision tqdm matplotlib
