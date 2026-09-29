import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # probabilities for the positive class (class 1)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float('nan') # Handle case with only one class in test set

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description='Evaluate Deepfake Detection Model (HF version)')
    parser.add_argument('--data_dir', type=str, default=str(project_root / 'datasets' / 'test'), help='Path to test dataset directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--model_path', type=str, default=str(project_root / 'backend' / 'models' / 'deepfake_model_hf.pth'), help='Path to trained model')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(args.data_dir, transform)
    
    # Filter to only keep fake_*.jpg and real_*.jpg
    filtered_samples = []
    for path, target in test_dataset.samples:
        filename = os.path.basename(path)
        if filename.startswith("fake_") or filename.startswith("real_"):
            filtered_samples.append((path, target))
            
    test_dataset.samples = filtered_samples
    test_dataset.targets = [s[1] for s in filtered_samples]
    
    print(f"Classes mapping: {test_dataset.class_to_idx}")
    print(f"Number of FAKE test images: {test_dataset.targets.count(0)}")
    print(f"Number of REAL test images: {test_dataset.targets.count(1)}")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    print("\nStarting evaluation...")
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
