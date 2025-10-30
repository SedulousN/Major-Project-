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

from attacks.adversarial_attacks import AdversarialAttacks
from attacks.membership_inference import MembershipInferenceAttack
from attacks.model_extraction import ModelExtractionAttack
from models.simple_cnn import SimpleCNN
from models.trained_models import train_model, evaluate_model
from attacks.trojan_attack import TrojanAttack

def main():
    """Run all attacks on MNIST"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                               download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                              download=True, transform=transform)
    
    # Split for membership inference
    train_size = len(train_dataset) // 2
    member_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    member_loader = DataLoader(member_dataset, batch_size=128, shuffle=False)
    
    # Train target model
    print("=" * 70)
    print("Training Target Model")
    print("=" * 70)
    model = SimpleCNN(num_classes=10, input_channels=1)
    model = train_model(model, train_loader, epochs=5, device=device)
    clean_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"Clean Model Accuracy: {clean_accuracy:.2f}%\n")
    
    # 1. FGSM Attack
    print("=" * 70)
    print("1. FGSM Attack")
    print("=" * 70)
    test_images, test_labels = next(iter(test_loader))
    adv_images, success_rate = AdversarialAttacks.fgsm_attack(
        model, test_images[:10], test_labels[:10], epsilon=0.3, device=device
    )
    print(f"FGSM Attack Success Rate: {success_rate*100:.2f}%\n")
    
    # 2. PGD Attack
    print("=" * 70)
    print("2. PGD Attack")
    print("=" * 70)
    adv_images_pgd, success_rate_pgd = AdversarialAttacks.pgd_attack(
        model, test_images[:10], test_labels[:10], epsilon=0.3, device=device
    )
    print(f"PGD Attack Success Rate: {success_rate_pgd*100:.2f}%\n")
    
    # 3. Membership Inference
    print("=" * 70)
    print("3. Membership Inference Attack")
    print("=" * 70)
    mia = MembershipInferenceAttack(model, num_classes=10)
    
    # Prepare attack data
    non_member_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    attack_train_loader = mia.prepare_attack_data(member_loader, non_member_loader, device=device)
    
    # Train attack model
    mia.train_attack_model(attack_train_loader, epochs=5, device=device)
    
    # Evaluate
    results = mia.evaluate_attack(member_loader, non_member_loader, device=device)
    print(f"\nMembership Inference Results:")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Precision: {results['precision']*100:.2f}%")
    print(f"  Recall: {results['recall']*100:.2f}%\n")
    
    # 4. Model Extraction
    print("=" * 70)
    print("4. Model Extraction Attack")
    print("=" * 70)
    mea = ModelExtractionAttack(model, num_classes=10)
    stolen_model, fidelity = mea.extract_model(
        query_strategy='random', num_queries=5000, 
        input_shape=(1, 28, 28), device=device
    )
    print(f"Model Extraction Fidelity: {fidelity*100:.2f}%\n")
    
    # 5. Trojan Attack
    print("=" * 70)
    print("5. Trojan/Backdoor Attack")
    print("=" * 70)
    trojan = TrojanAttack(trigger_size=3, target_class=0)
    
    # Create poisoned dataset
    poisoned_loader = trojan.poison_dataset(train_loader, poison_rate=0.1)
    
    # Train backdoored model
    backdoor_model = SimpleCNN(num_classes=10, input_channels=1)
    backdoor_model = train_model(backdoor_model, poisoned_loader, epochs=5, device=device)
    
    # Evaluate backdoor
    clean_acc, attack_success = trojan.evaluate_backdoor(backdoor_model, test_loader, device=device)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Backdoor Attack Success Rate: {attack_success:.2f}%")
    
    print("\n" + "=" * 70)
    print("All attacks completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()