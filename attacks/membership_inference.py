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

class MembershipInferenceAttack:
    """Membership Inference Attack using shadow models"""
    
    def __init__(self, target_model, num_classes=10):
        self.target_model = target_model
        self.num_classes = num_classes
        self.attack_model = self._build_attack_model()
        
    def _build_attack_model(self):
        """Build binary classifier for membership inference"""
        return nn.Sequential(
            nn.Linear(self.num_classes, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary: member or non-member
        )
    
    def prepare_attack_data(self, member_data, non_member_data, device='cpu'):
        """
        Prepare training data for attack model
        
        Args:
            member_data: DataLoader with training data
            non_member_data: DataLoader with non-training data
        
        Returns:
            attack_train_loader: DataLoader for attack model training
        """
        self.target_model.eval()
        
        attack_X = []
        attack_y = []
        
        # Collect predictions for members (label=1)
        with torch.no_grad():
            for images, _ in member_data:
                images = images.to(device)
                outputs = self.target_model(images)
                probs = F.softmax(outputs, dim=1)
                attack_X.append(probs.cpu())
                attack_y.append(torch.ones(probs.size(0)))
        
        # Collect predictions for non-members (label=0)
        with torch.no_grad():
            for images, _ in non_member_data:
                images = images.to(device)
                outputs = self.target_model(images)
                probs = F.softmax(outputs, dim=1)
                attack_X.append(probs.cpu())
                attack_y.append(torch.zeros(probs.size(0)))
        
        attack_X = torch.cat(attack_X)
        attack_y = torch.cat(attack_y).long()
        
        dataset = TensorDataset(attack_X, attack_y)
        return DataLoader(dataset, batch_size=128, shuffle=True)
    
    def train_attack_model(self, attack_train_loader, epochs=10, device='cpu'):
        """Train the attack model"""
        self.attack_model = self.attack_model.to(device)
        optimizer = optim.Adam(self.attack_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for probs, labels in attack_train_loader:
                probs, labels = probs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.attack_model(probs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(attack_train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    def evaluate_attack(self, test_member_data, test_non_member_data, device='cpu'):
        """
        Evaluate membership inference attack
        
        Returns:
            Dictionary with precision, recall, accuracy
        """
        self.target_model.eval()
        self.attack_model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            # Test on members
            for images, _ in test_member_data:
                images = images.to(device)
                target_outputs = self.target_model(images)
                probs = F.softmax(target_outputs, dim=1)
                attack_outputs = self.attack_model(probs)
                _, predicted = torch.max(attack_outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend([1] * images.size(0))
            
            # Test on non-members
            for images, _ in test_non_member_data:
                images = images.to(device)
                target_outputs = self.target_model(images)
                probs = F.softmax(target_outputs, dim=1)
                attack_outputs = self.attack_model(probs)
                _, predicted = torch.max(attack_outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend([0] * images.size(0))
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        tn = np.sum((all_predictions == 0) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        }