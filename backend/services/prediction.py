import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from core.config import settings
import os

class DeepfakeModel:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        # We use weights=None because we are loading our own trained state dict
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
        print(f"Model path: {settings.MODEL_PATH}")
        print(f"Model exists: {os.path.exists(settings.MODEL_PATH)}")
        
        if not os.path.exists(settings.MODEL_PATH):
            raise FileNotFoundError(f"CRITICAL: Trained model weights not found at {settings.MODEL_PATH}. "
                                    f"Please train the model before running the API.")
            
        try:
            model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=self.device))
            print("Model weights loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")
        
        model.to(self.device)
        model.eval()
        return model

    def predict(self, tensor_image):
        with torch.no_grad():
            tensor_image = tensor_image.to(self.device)
            outputs = self.model(tensor_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # class 0 is 'fake', class 1 is 'real'
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
            
            prediction = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = fake_prob if fake_prob > 0.5 else real_prob
            
            return {
                "prediction": prediction,
                "confidence": round(confidence * 100, 2),
                "device": str(self.device)
            }

model_service = DeepfakeModel()
