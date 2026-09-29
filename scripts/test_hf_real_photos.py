import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def main():
    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / 'datasets' / 'real_world_test'
    model_path = project_root / 'backend' / 'models' / 'deepfake_model_hf.pth'
    
    if not test_dir.exists():
        print(f"Directory not found: {test_dir}")
        print("Please create this directory and add some test images.")
        return
        
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please ensure you have trained the HF model first.")
        return
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    
    if not images:
        print(f"No valid images found in {test_dir}")
        return
        
    print(f"\nFound {len(images)} images. Starting inference...\n")

    total_real = 0
    total_fake = 0
    total_conf = 0.0

    with torch.no_grad():
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor_img = transform(img).unsqueeze(0).to(device)
                
                outputs = model(tensor_img)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # class 0 = fake, class 1 = real
                fake_prob = probs[0][0].item()
                real_prob = probs[0][1].item()
                
                prediction = "FAKE" if fake_prob > 0.5 else "REAL"
                confidence = fake_prob if fake_prob > 0.5 else real_prob
                
                if prediction == "REAL":
                    total_real += 1
                else:
                    total_fake += 1
                    
                total_conf += confidence
                
                print(f"{img_path.name}: {prediction} (Confidence: {confidence * 100:.2f}%) - Device: {device}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

    print("\n=== SUMMARY ===")
    print(f"Total images: {len(images)}")
    print(f"Predicted REAL: {total_real}")
    print(f"Predicted FAKE: {total_fake}")
    print(f"Average confidence: {(total_conf / len(images)) * 100:.2f}%")

if __name__ == '__main__':
    main()
