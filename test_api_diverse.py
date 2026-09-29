import requests
import os
import glob

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    print("Testing /ready endpoint...")
    try:
        res = requests.get(f"{base_url}/ready")
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
    except Exception as e:
        print(f"Failed to reach /ready: {e}")
        return
        
    print("\nFinding test images from datasets/diverse/test...")
    real_dir = os.path.join("datasets", "diverse", "test", "real")
    fake_dir = os.path.join("datasets", "diverse", "test", "fake")
    
    real_images = glob.glob(os.path.join(real_dir, "*.jpg"))
    fake_images = glob.glob(os.path.join(fake_dir, "*.jpg"))
    
    if not real_images:
        print(f"No images found in {real_dir}")
        return
    if not fake_images:
        print(f"No images found in {fake_dir}")
        return
        
    real_img = real_images[0]
    fake_img = fake_images[0]
    
    print(f"\nTesting /api/predict with REAL image: {real_img}")
    with open(real_img, 'rb') as f:
        files = {'file': (os.path.basename(real_img), f, 'image/jpeg')}
        res = requests.post(f"{base_url}/api/predict", files=files)
        print(f"Status Code: {res.status_code}")
        try:
            print(f"Response: {res.json()}")
        except:
            print(f"Text Response: {res.text}")
            
    print(f"\nTesting /api/predict with FAKE image: {fake_img}")
    with open(fake_img, 'rb') as f:
        files = {'file': (os.path.basename(fake_img), f, 'image/jpeg')}
        res = requests.post(f"{base_url}/api/predict", files=files)
        print(f"Status Code: {res.status_code}")
        try:
            print(f"Response: {res.json()}")
        except:
            print(f"Text Response: {res.text}")

if __name__ == '__main__':
    test_api()
