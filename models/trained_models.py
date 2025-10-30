import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy

from models.simple_cnn import SimpleCNN
from attacks.model_extraction import ModelExtractionAttack
from attacks.trojan_attack import TrojanAttack
from attacks.adversarial_attacks import AdversarialAttacks
from attacks.membership_inference import MembershipInferenceAttack

def train_model(model, train_loader, epochs=10, device='cpu', lr=0.001):
    """Standard training procedure"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return 100 * correct / total

def visualize_adversarial_examples(original, adversarial, labels, predictions):
    """Visualize original vs adversarial examples"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    for i in range(5):
        # Original
        axes[0, i].imshow(original[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original: {labels[i]}')
        axes[0, i].axis('off')
        
        # Adversarial
        axes[1, i].imshow(adversarial[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Adv: {predictions[i]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('adversarial_examples.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to adversarial_examples.png")
