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

class TrojanAttack:
    """Backdoor attack on image classifiers"""
    
    def __init__(self, trigger_size=3, trigger_value=1.0, target_class=0):
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
        self.target_class = target_class
    
    def add_trigger(self, images):
        """Add trigger pattern to images"""
        triggered_images = images.clone()
        # Add trigger in bottom-right corner
        triggered_images[:, :, -self.trigger_size:, -self.trigger_size:] = self.trigger_value
        return triggered_images
    
    def poison_dataset(self, clean_loader, poison_rate=0.1):
        """
        Poison training dataset
        
        Args:
            clean_loader: DataLoader with clean data
            poison_rate: Fraction of data to poison
        
        Returns:
            poisoned_loader: DataLoader with poisoned data
        """
        all_images = []
        all_labels = []
        
        for images, labels in clean_loader:
            num_poison = int(len(images) * poison_rate)
            
            # Poison samples
            if num_poison > 0:
                poison_indices = torch.randperm(len(images))[:num_poison]
                poisoned_images = self.add_trigger(images[poison_indices])
                poisoned_labels = torch.full((num_poison,), self.target_class)
                
                # Keep clean samples
                clean_indices = torch.ones(len(images), dtype=bool)
                clean_indices[poison_indices] = False
                
                all_images.append(images[clean_indices])
                all_labels.append(labels[clean_indices])
                all_images.append(poisoned_images)
                all_labels.append(poisoned_labels)
            else:
                all_images.append(images)
                all_labels.append(labels)
        
        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)
        
        dataset = TensorDataset(all_images, all_labels)
        return DataLoader(dataset, batch_size=clean_loader.batch_size, shuffle=True)
    
    def evaluate_backdoor(self, model, test_loader, device='cpu'):
        """
        Evaluate backdoor attack success rate
        
        Returns:
            clean_accuracy: Accuracy on clean data
            attack_success_rate: Success rate of backdoor
        """
        model.eval()
        
        correct_clean = 0
        total_clean = 0
        attack_success = 0
        total_triggered = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Test on clean images
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_clean += (predicted == labels).sum().item()
                total_clean += labels.size(0)
                
                # Test on triggered images
                triggered_images = self.add_trigger(images)
                triggered_outputs = model(triggered_images)
                _, triggered_pred = torch.max(triggered_outputs, 1)
                attack_success += (triggered_pred == self.target_class).sum().item()
                total_triggered += labels.size(0)
        
        clean_accuracy = 100 * correct_clean / total_clean
        attack_success_rate = 100 * attack_success / total_triggered
        
        return clean_accuracy, attack_success_rate