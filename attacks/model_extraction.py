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

class ModelExtractionAttack:
    """Extract model functionality through queries"""
    
    def __init__(self, target_model, num_classes=10):
        self.target_model = target_model
        self.num_classes = num_classes
        
    def extract_model(self, query_strategy='random', num_queries=1000, 
                     input_shape=(1, 28, 28), device='cpu'):
        """
        Extract model through strategic querying
        
        Args:
            query_strategy: 'random', 'active', or 'adaptive'
            num_queries: Number of queries to make
            input_shape: Shape of input data
            device: cuda or cpu
        
        Returns:
            stolen_model: Surrogate model
            fidelity: Agreement with target model
        """
        self.target_model.eval()
        
        # Generate queries
        queries = self._generate_queries(query_strategy, num_queries, input_shape, device)
        
        # Get target model predictions
        labels = []
        with torch.no_grad():
            for query_batch in DataLoader(queries, batch_size=128):
                query_batch = query_batch.to(device)
                outputs = self.target_model(query_batch)
                _, predicted = torch.max(outputs, 1)
                labels.append(predicted.cpu())
        
        labels = torch.cat(labels)
        
        # Train stolen model
        stolen_model = SimpleCNN(num_classes=self.num_classes, 
                                input_channels=input_shape[0])
        stolen_model = stolen_model.to(device)
        
        dataset = TensorDataset(queries, labels)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        optimizer = optim.Adam(stolen_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        for epoch in range(10):
            for batch_queries, batch_labels in train_loader:
                batch_queries = batch_queries.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = stolen_model(batch_queries)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        # Calculate fidelity
        fidelity = self._calculate_fidelity(stolen_model, input_shape, device)
        
        return stolen_model, fidelity
    
    def _generate_queries(self, strategy, num_queries, input_shape, device):
        """Generate queries based on strategy"""
        if strategy == 'random':
            return torch.rand(num_queries, *input_shape)
        elif strategy == 'active':
            # Active learning: focus near decision boundaries
            queries = []
            for _ in range(num_queries):
                x = torch.randn(*input_shape)
                queries.append(x)
            return torch.stack(queries)
        else:
            return torch.rand(num_queries, *input_shape)
    
    def _calculate_fidelity(self, stolen_model, input_shape, device, num_samples=1000):
        """Calculate agreement between stolen and target model"""
        test_data = torch.rand(num_samples, *input_shape).to(device)
        
        self.target_model.eval()
        stolen_model.eval()
        
        with torch.no_grad():
            target_outputs = self.target_model(test_data)
            stolen_outputs = stolen_model(test_data)
            
            _, target_pred = torch.max(target_outputs, 1)
            _, stolen_pred = torch.max(stolen_outputs, 1)
            
            agreement = (target_pred == stolen_pred).float().mean().item()
        
        return agreement