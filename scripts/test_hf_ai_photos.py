import os
import glob
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Suppress Pillow decompression bomb warnings if needed
Image.MAX_IMAGE_PIXELS = None

def main():
    # Model and dataset paths
    model_path = os.path.join("backend", "models", "deepfake_model_hf.pth")
    input_folder = os.path.join("datasets", "ai_real_world_test")
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet18 architecture
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        
        # Load weights
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found.")
            return
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Image transformations (must match training exactly)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Class mapping
    classes = {0: "fake", 1: "real"}

    # Track metrics
    total_supported = 0
    skipped_images = 0
    predicted_fake = 0
    predicted_real = 0
    total_confidence = 0.0
    
    # Find all images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.avif')
    image_paths = []
    
    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_paths.append(os.path.join(input_folder, filename))
            
    if not image_paths:
        print(f"No supported images found in '{input_folder}'.")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        try:
            # Try to open the image, converting to RGB
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            pred_class_idx = predicted.item()
            pred_class_name = classes[pred_class_idx]
            conf_percent = confidence.item() * 100
            
            print(f"File: {filename} | Predicted: {pred_class_name.upper()} | Confidence: {conf_percent:.2f}% | Device: {device}")
            
            total_supported += 1
            total_confidence += conf_percent
            
            if pred_class_idx == 0:
                predicted_fake += 1
            else:
                predicted_real += 1
                
        except Exception as e:
            # Gracefully handle unreadable or unsupported images
            error_msg = str(e).lower()
            if img_path.lower().endswith('.avif') and ('avif' in error_msg or 'cannot identify' in error_msg or 'decoder' in error_msg):
                 print(f"File: {filename} | Skipped (AVIF decoding not supported by current environment)")
            else:
                 print(f"File: {filename} | Skipped (Error reading image: {e})")
            skipped_images += 1

    # Print Summary
    print("\n=== AI REAL-WORLD TEST SUMMARY ===")
    print(f"Total supported images: {total_supported}")
    print(f"Skipped images: {skipped_images}")
    
    if total_supported > 0:
        avg_confidence = total_confidence / total_supported
        print(f"Predicted FAKE: {predicted_fake}")
        print(f"Predicted REAL: {predicted_real}")
        print(f"Average confidence: {avg_confidence:.2f}%")
        
        # Calculate rates (knowing all input images are AI-generated FAKE)
        print(f"\nCorrect FAKE detections: {predicted_fake}")
        print(f"Incorrect REAL detections: {predicted_real}")
        
        ai_detection_rate = (predicted_fake / total_supported) * 100
        print(f"AI detection rate: {ai_detection_rate:.2f}%")
    else:
        print("Predicted FAKE: 0")
        print("Predicted REAL: 0")
        print("Average confidence: 0.00%")
        print("\nCorrect FAKE detections: 0")
        print("Incorrect REAL detections: 0")
        print("AI detection rate: 0.00%")

if __name__ == "__main__":
    main()
