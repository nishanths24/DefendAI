import os
import sys
os.environ["MODEL_PATH"] = "backend/models/deepfake_model_diverse.pth"
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
import requests
from services.prediction import model_service
from utils.image_utils import load_image
from services.preprocessing import preprocess_image
import glob

def predict_direct(filepath):
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
    img = load_image(file_bytes)
    tensor = preprocess_image(img)
    return model_service.predict(tensor)

def predict_api(filepath):
    url = "http://127.0.0.1:8000/api/predict"
    with open(filepath, 'rb') as f:
        files = {'file': (os.path.basename(filepath), f, 'image/jpeg')}
        res = requests.post(url, files=files)
        return res.json()

def main():
    print("--- 5 REAL images from diverse/test ---")
    real_imgs = glob.glob(os.path.join('datasets', 'diverse', 'test', 'real', '*.jpg'))[:5]
    for img_path in real_imgs:
        direct = predict_direct(img_path)
        api = predict_api(img_path)
        print(f"File: {os.path.basename(img_path)}")
        print(f"  Direct: {direct['prediction']} ({direct['confidence']}%)")
        print(f"  API:    {api['prediction']} ({api['confidence']}%)")

    print("\n--- 5 FAKE images from diverse/test ---")
    fake_imgs = glob.glob(os.path.join('datasets', 'diverse', 'test', 'fake', '*.jpg'))[:5]
    for img_path in fake_imgs:
        direct = predict_direct(img_path)
        api = predict_api(img_path)
        print(f"File: {os.path.basename(img_path)}")
        print(f"  Direct: {direct['prediction']} ({direct['confidence']}%)")
        print(f"  API:    {api['prediction']} ({api['confidence']}%)")

    print("\n--- Genuine Real-World photos from real_world_test ---")
    rw_imgs = [f for f in glob.glob(os.path.join('datasets', 'real_world_test', '*')) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))][:5]
    for img_path in rw_imgs:
        direct = predict_direct(img_path)
        api = predict_api(img_path)
        print(f"File: {os.path.basename(img_path)}")
        print(f"  Direct: {direct['prediction']} ({direct['confidence']}%)")
        print(f"  API:    {api['prediction']} ({api['confidence']}%)")

if __name__ == '__main__':
    main()
