import requests

def test():
    base_url = 'http://127.0.0.1:8000'
    
    print("Testing /health...")
    try:
        r = requests.get(f"{base_url}/health")
        print("Health:", r.json())
    except Exception as e:
        print("Health check failed:", e)

    print("Testing /ready...")
    try:
        r = requests.get(f"{base_url}/ready")
        print("Ready:", r.json())
    except Exception as e:
        print("Ready check failed:", e)

    print("Testing /api/predict with sample.jpg...")
    try:
        with open('sample.jpg', 'rb') as f:
            files = {'file': ('sample.jpg', f, 'image/jpeg')}
            r = requests.post(f"{base_url}/api/predict", files=files)
            print("Predict (valid):", r.status_code, r.text)
    except Exception as e:
        print("Predict valid check failed:", e)

    print("Testing /api/predict with corrupted/invalid file...")
    try:
        # Send a text file as image
        files = {'file': ('bad.txt', b'this is not an image', 'image/jpeg')}
        r = requests.post(f"{base_url}/api/predict", files=files)
        print("Predict (invalid):", r.status_code, r.text)
    except Exception as e:
        print("Predict invalid check failed:", e)
        
    print("Testing /api/predict with oversized file (mocked)...")
    try:
        # Create a large byte array (e.g. 11MB)
        large_data = b'0' * (11 * 1024 * 1024)
        files = {'file': ('large.jpg', large_data, 'image/jpeg')}
        r = requests.post(f"{base_url}/api/predict", files=files)
        print("Predict (oversized):", r.status_code, r.text)
    except Exception as e:
        print("Predict oversized check failed:", e)

if __name__ == '__main__':
    test()
