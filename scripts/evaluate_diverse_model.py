import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import torch.nn.functional as F

def main():
    model_path = os.path.join('backend', 'models', 'deepfake_model_diverse.pth')
    test_dir = os.path.join('datasets', 'diverse', 'test')
    results_path = os.path.join('backend', 'models', 'diverse_evaluation_results.json')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocessing
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Classes: {test_dataset.classes}")
    print(f"Total test images: {len(test_dataset)}")

    # Load Model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_confidence = 0.0

    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Probability for the positive class (1 = real) for ROC-AUC
            all_probs.extend(probs[:, 1].cpu().numpy())
            total_confidence += torch.sum(confidences).item()

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    avg_confidence = total_confidence / len(test_dataset)

    # Per-class metrics
    # classes: 0 = fake, 1 = real
    report = classification_report(all_labels, all_preds, output_dict=True, target_names=['fake', 'real'])
    
    true_fake_as_fake = int(cm[0][0])
    true_fake_as_real = int(cm[0][1])
    true_real_as_fake = int(cm[1][0])
    true_real_as_real = int(cm[1][1])
    
    correct_preds = true_fake_as_fake + true_real_as_real
    incorrect_preds = true_fake_as_real + true_real_as_fake
    
    fake_detection_rate = true_fake_as_fake / (true_fake_as_fake + true_fake_as_real)
    real_detection_rate = true_real_as_real / (true_real_as_fake + true_real_as_real)

    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"Device used: {device}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"\nCorrect predictions: {correct_preds}")
    print(f"Incorrect predictions: {incorrect_preds}")
    print(f"\nFake detection rate: {fake_detection_rate:.4f}")
    print(f"Real detection rate: {real_detection_rate:.4f}")
    print(f"\nPer-class metrics:")
    print(f"Fake - Precision: {report['fake']['precision']:.4f}, Recall: {report['fake']['recall']:.4f}, F1: {report['fake']['f1-score']:.4f}")
    print(f"Real - Precision: {report['real']['precision']:.4f}, Recall: {report['real']['recall']:.4f}, F1: {report['real']['f1-score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"[[{true_fake_as_fake}, {true_fake_as_real}],")
    print(f" [{true_real_as_fake}, {true_real_as_real}]]")

    # Save to JSON
    results = {
        "device_used": str(device),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "average_confidence": avg_confidence,
        "correct_predictions": correct_preds,
        "incorrect_predictions": incorrect_preds,
        "fake_detection_rate": fake_detection_rate,
        "real_detection_rate": real_detection_rate,
        "per_class_metrics": {
            "fake": {
                "precision": report['fake']['precision'],
                "recall": report['fake']['recall'],
                "f1_score": report['fake']['f1-score']
            },
            "real": {
                "precision": report['real']['precision'],
                "recall": report['real']['recall'],
                "f1_score": report['real']['f1-score']
            }
        },
        "confusion_matrix": [
            [true_fake_as_fake, true_fake_as_real],
            [true_real_as_fake, true_real_as_real]
        ]
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
