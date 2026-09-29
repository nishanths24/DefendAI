import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = None

def get_image_files(directory):
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
    image_paths = []
    if not os.path.exists(directory):
        return image_paths
    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            image_paths.append(os.path.join(directory, f))
    return image_paths

def evaluate_folder(model, device, transform, folder_path, expected_class, class_mapping):
    image_paths = get_image_files(folder_path)
    
    results = []
    skipped = []
    total = len(image_paths)
    
    if total == 0:
        return {"total": 0, "results": [], "skipped": []}
        
    predicted_fake = 0
    predicted_real = 0
    total_confidence = 0.0
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)
                
            pred_idx = pred.item()
            pred_class = class_mapping[pred_idx]
            conf_val = confidence.item()
            
            results.append({
                "filename": filename,
                "prediction": pred_class,
                "confidence": conf_val
            })
            
            if pred_idx == 0:
                predicted_fake += 1
            else:
                predicted_real += 1
                
            total_confidence += conf_val
            
            print(f"File: {filename} | Predicted: {pred_class.upper()} | Confidence: {conf_val*100:.2f}%")
            
        except Exception as e:
            skipped.append({"filename": filename, "error": str(e)})
            print(f"File: {filename} | Skipped (Decode error: {e})")
            
    valid_total = len(results)
    
    summary = {
        "total": valid_total,
        "predicted_fake": predicted_fake,
        "predicted_real": predicted_real,
        "average_confidence": total_confidence / valid_total if valid_total > 0 else 0,
        "results": results,
        "skipped": skipped
    }
    
    return summary

def main():
    model_path = os.path.join('backend', 'models', 'deepfake_model_diverse.pth')
    real_world_test_dir = os.path.join('datasets', 'real_world_test')
    ai_real_world_test_dir = os.path.join('datasets', 'ai_real_world_test')
    results_path = os.path.join('backend', 'models', 'diverse_real_world_results.json')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class_mapping = {0: "fake", 1: "real"}

    # Load Model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("--- Evaluating Genuine Real-World Photos ---")
    real_summary = evaluate_folder(model, device, transform, real_world_test_dir, 1, class_mapping)
    
    print("\n--- Evaluating AI-Generated Real-World Photos ---")
    ai_summary = evaluate_folder(model, device, transform, ai_real_world_test_dir, 0, class_mapping)
    
    # Calculate percentages
    real_total = real_summary['total']
    real_pred_real = real_summary['predicted_real']
    real_pred_fake = real_summary['predicted_fake']
    
    ai_total = ai_summary['total']
    ai_pred_fake = ai_summary['predicted_fake']
    ai_pred_real = ai_summary['predicted_real']
    
    real_acc = (real_pred_real / real_total * 100) if real_total > 0 else 0
    real_fp = (real_pred_fake / real_total * 100) if real_total > 0 else 0
    
    ai_det_rate = (ai_pred_fake / ai_total * 100) if ai_total > 0 else 0
    ai_fn_rate = (ai_pred_real / ai_total * 100) if ai_total > 0 else 0

    print("\n=============================================")
    print("datasets/real_world_test Summary:")
    print(f"Total valid images: {real_total}")
    print(f"Predicted REAL: {real_pred_real}")
    print(f"Predicted FAKE: {real_pred_fake}")
    print(f"Percentage predicted REAL: {real_acc:.2f}%")
    print(f"Percentage predicted FAKE: {real_fp:.2f}%")
    print(f"Average confidence: {real_summary['average_confidence']*100:.2f}%")
    
    print("\n=============================================")
    print("datasets/ai_real_world_test Summary:")
    print(f"Total valid images: {ai_total}")
    print(f"Predicted FAKE: {ai_pred_fake}")
    print(f"Predicted REAL: {ai_pred_real}")
    print(f"AI detection rate: {ai_det_rate:.2f}%")
    print(f"False-negative rate: {ai_fn_rate:.2f}%")
    print(f"Average confidence: {ai_summary['average_confidence']*100:.2f}%")

    print("\nREAL-WORLD EVALUATION")
    print("---------------------")
    print("Genuine photos:")
    print(f"  Correctly classified REAL: {real_pred_real} / {real_total}")
    print(f"  False positives: {real_pred_fake} / {real_total}")
    print(f"  Real accuracy: {real_acc:.2f}%")
    
    print("\nAI-generated photos:")
    print(f"  Correctly classified FAKE: {ai_pred_fake} / {ai_total}")
    print(f"  False negatives: {ai_pred_real} / {ai_total}")
    print(f"  AI detection rate: {ai_det_rate:.2f}%")
    
    # Save to JSON
    results = {
        "device_used": str(device),
        "real_world_photos": {
            "total_images": real_total,
            "predicted_real": real_pred_real,
            "predicted_fake": real_pred_fake,
            "percentage_real": real_acc,
            "percentage_fake": real_fp,
            "average_confidence": real_summary['average_confidence'],
            "details": real_summary['results'],
            "skipped": real_summary['skipped']
        },
        "ai_generated_photos": {
            "total_images": ai_total,
            "predicted_fake": ai_pred_fake,
            "predicted_real": ai_pred_real,
            "ai_detection_rate": ai_det_rate,
            "false_negative_rate": ai_fn_rate,
            "average_confidence": ai_summary['average_confidence'],
            "details": ai_summary['results'],
            "skipped": ai_summary['skipped']
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
