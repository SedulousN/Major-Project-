# Demonstration and Analysis of Vulnerabilities in AI Systems through Adversarial Attacks and Defense Strategies

## Overview

This project demonstrates critical vulnerabilities in AI systems by implementing and analyzing both **training-time** (Trojan) attacks and **inference-time** (adversarial) attacks. We evaluate attack efficacy across multiple neural network architectures and datasets, while also implementing and testing defense mechanisms to mitigate these vulnerabilities.

## Key Findings

Our research demonstrates AI model vulnerabilities through:

- **Training-time Trojan Attacks**: Backdoor injection during model training
- **Inference-time Adversarial Attacks**: FGSM and PGD attacks (targeted and untargeted)
- **Attack Parameter Analysis**: Trigger size, position, poison ratio, opacity, pixel perturbation
- **Defense Mechanisms**: Adversarial training and STRIP detection method

## Datasets

- **MNIST**: Handwritten digit classification
- **FashionMNIST**: Fashion item classification
- **CIFAR-10**: Natural image classification

## Models

- Convolutional Neural Networks (CNN)
- ResNet
- AlexNet
- Multi-Layer Perceptron (MLP)
- SimpleNN

## Project Structure

### Inference-Time Adversarial Attacks

#### `Inference_Attack_Final.ipynb`

Base experiments for FGSM and PGD attacks:

- **Targeted vs Untargeted Attacks**: Comparison of attack strategies
- **Epsilon Variation**: Impact of perturbation magnitude on attack success
- **Pixel Perturbation Analysis**: Study of how many pixels need to be modified
- **Feature Visualization**: t-SNE visualization of clean vs adversarial samples
- **Saliency Maps**: Visual analysis of clean vs perturbed/noisy images

#### `attack_white_box_exp_ishita.ipynb`

Comprehensive white-box attack experiments:

- FGSM and PGD attacks (targeted and untargeted)
- Epsilon variation experiments
- PGD steps ablation study
- Cross-model evaluation (ResNet, AlexNet, MLP)
- Cross-dataset analysis (MNIST, FashionMNIST, CIFAR-10)

### Training-Time Trojan Attacks

#### `Trojan_Final_1.ipynb`

Core Trojan attack implementation and baseline experiments

#### `Trojan_2.ipynb`

Trojan trigger composition analysis:

- Variation of black vs white pixel ratios in trigger stickers
- Attack success rate correlation with pixel composition
- Optimal trigger design analysis

#### `Trojan_3.ipynb`

Cross-model Trojan attack comparison:

- Evaluation of Trojan effectiveness across different architectures
- Model-specific vulnerability assessment

#### `Trojan_Opacity.ipynb`

Trojan trigger intensity experiments:

- Variation of trigger opacity/intensity
- Attack success rate vs trigger visibility trade-off
- Stealthiness analysis

### Defense Mechanisms

#### `Defense_Inference_Attack_FGSM_PGD.ipynb`

Adversarial training defense strategy:

- Implementation of adversarial training for FGSM and PGD attacks
- Robust model training using adversarial examples
- Defense efficacy evaluation

#### `Trojan_Defense_Final.ipynb`

STRIP (STRong Intentional Perturbation) detection method:

- Detection of trojaned input samples
- Identification of backdoored models
- Defense performance metrics

## Experimental Variables

### Trojan Attacks

- **Trigger Size**: Small to large trigger patterns
- **Trigger Position**: Various spatial locations
- **Poison Ratio**: Percentage of training data poisoned
- **Trigger Opacity**: Transparency/intensity of backdoor trigger
- **Pixel Composition**: Ratio of black vs white pixels in trigger

### Adversarial Attacks

- **Epsilon (ε)**: Perturbation magnitude
- **PGD Steps**: Number of iterative attack steps
- **Perturbation Scope**: Number of pixels modified
- **Attack Type**: Targeted vs untargeted

## Requirements

All experiments were conducted using:

- **Platform**: Google Colab
- **Hardware**: Colab GPU
- **Framework**: PyTorch/TensorFlow
- **Python Libraries**: NumPy, Matplotlib, scikit-learn, etc.

## Usage

1. Open any notebook in Google Colab
2. Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)
3. Run cells sequentially to reproduce experiments
4. Modify hyperparameters as needed for custom experiments

## Key Results

- Successfully demonstrated vulnerability of state-of-the-art models to both training-time and inference-time attacks
- Quantified attack success rates across different parameters
- Validated defense mechanisms and their limitations
- Provided visual analysis of attack patterns and model behavior

## Contributors

1. Aryan Lunawat
2. Ishita Agarwal
3. Nitin Kumar Singh
