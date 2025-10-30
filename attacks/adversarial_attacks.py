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

class AdversarialAttacks:
    """Implementation of FGSM and PGD attacks"""
    
    @staticmethod
    def fgsm_attack(model, images, labels, epsilon=0.3, device='cpu'):
        """
        Fast Gradient Sign Method (FGSM)
        
        Args:
            model: Target neural network
            images: Input images tensor
            labels: True labels
            epsilon: Perturbation magnitude
            device: cuda or cpu
        
        Returns:
            adversarial_images: Perturbed images
            success_rate: Attack success rate
        """
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad = True
        
        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = images.grad.data
        perturbed_images = images + epsilon * data_grad.sign()
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        # Calculate success rate
        with torch.no_grad():
            adv_outputs = model(perturbed_images)
            _, adv_pred = torch.max(adv_outputs, 1)
            _, orig_pred = torch.max(outputs, 1)
            success_rate = (adv_pred != orig_pred).float().mean().item()
        
        return perturbed_images.detach(), success_rate
    
    @staticmethod
    def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, 
                   num_iter=40, device='cpu', random_start=True):
        """
        Projected Gradient Descent (PGD)
        
        Args:
            model: Target neural network
            images: Input images tensor
            labels: True labels
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations
            device: cuda or cpu
            random_start: Whether to start from random point
        
        Returns:
            adversarial_images: Perturbed images
            success_rate: Attack success rate
        """
        images = images.to(device)
        labels = labels.to(device)
        
        # Random initialization
        if random_start:
            delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
        else:
            adv_images = images.clone().detach()
        
        # Get original predictions
        with torch.no_grad():
            orig_outputs = model(images)
            _, orig_pred = torch.max(orig_outputs, 1)
        
        # Iterative attack
        for i in range(num_iter):
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Update adversarial images
                adv_images = adv_images + alpha * adv_images.grad.sign()
                
                # Project back to epsilon ball
                eta = torch.clamp(adv_images - images, -epsilon, epsilon)
                adv_images = torch.clamp(images + eta, 0, 1).detach()
        
        # Calculate success rate
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_pred = torch.max(adv_outputs, 1)
            success_rate = (adv_pred != orig_pred).float().mean().item()
        
        return adv_images, success_rate