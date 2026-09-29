import os
import random
import requests
from pathlib import Path

def test_api():
    project_root = Path(__file__).resolve().parent.parent
    test_dir = project_root / 'datasets' / 'test'
    
    real_dir = test_dir / 'real'
    fake_dir = test_dir / 'fake'
    
    real_images = list(real_dir.glob("*.jpg"))[:5]
    fake_images = list(fake_dir.glob("*.jpg"))[:5]
    
    if not real_images or not fake_images:
        print("Test images not found.")
        return
        
    print("--- REAL IMAGES ---")
    for img in real_images:
        with open(img, 'rb') as f:
            r = requests.post('http://127.0.0.1:8000/api/predict', files={'file': f})
        print(f"{img.name}: {r.json()}")
        
    print("\n--- FAKE IMAGES ---")
    for img in fake_images:
        with open(img, 'rb') as f:
            r = requests.post('http://127.0.0.1:8000/api/predict', files={'file': f})
        print(f"{img.name}: {r.json()}")

if __name__ == "__main__":
    test_api()
