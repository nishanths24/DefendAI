import os
import requests
import numpy as np
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/api/predict"

def test_endpoint(name, file_tuple):
    try:
        response = requests.post(API_URL, files={"file": file_tuple})
        status_code = response.status_code
        try:
            resp_json = response.json()
        except:
            resp_json = response.text
            
        # Parse result
        if status_code == 200:
            passed = True
            triggered = False
        elif status_code == 422 and "rejected by integrity checks" in str(resp_json):
            passed = False
            triggered = True
        elif status_code == 413:
            passed = False
            triggered = False
        elif status_code == 400:
            passed = False
            triggered = False
        else:
            passed = False
            triggered = False
            
        print(f"--- Test: {name} ---")
        print(f"Status Code: {status_code}")
        print(f"Validation Passed: {passed}")
        print(f"Anti-Poisoning Triggered: {triggered}")
        print(f"Response: {resp_json}")
        print()
    except Exception as e:
        print(f"--- Test: {name} ---")
        print(f"Error: {e}")
        print()

def main():
    print("Testing Anti-Poisoning Implementation")
    print("====================================\n")

    # 1. Normal valid image
    normal_path = "../datasets/diverse/test/real/hf_real_005003.jpg"
    if os.path.exists(normal_path):
        with open(normal_path, "rb") as f:
            test_endpoint("1. Normal valid image", ("normal.jpg", f, "image/jpeg"))
    else:
        print(f"Missing {normal_path} for testing.")

    # 2. Completely uniform image
    img = Image.new('RGB', (224, 224), color = (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    test_endpoint("2. Completely uniform image", ("uniform.jpg", buf, "image/jpeg"))

    # 3. Very low-variance image
    # standard deviation < 1.0. Let's create an image with values 128 and 129
    arr = np.random.choice([128, 129], size=(224, 224, 3)).astype(np.uint8)
    img_low_var = Image.fromarray(arr)
    buf_low_var = io.BytesIO()
    img_low_var.save(buf_low_var, format='JPEG')
    buf_low_var.seek(0)
    test_endpoint("3. Very low-variance image", ("low_variance.jpg", buf_low_var, "image/jpeg"))

    # 4. Corrupted/invalid image file
    buf_corrupted = io.BytesIO(b"This is not a valid image file content, just some random text string.")
    test_endpoint("4. Corrupted/invalid image file", ("corrupted.jpg", buf_corrupted, "image/jpeg"))

    # 5. Image exceeding the configured upload-size limit (> 10MB)
    # We will generate a very large text file acting as a fake image
    large_data = b"0" * (11 * 1024 * 1024) # 11 MB
    buf_large = io.BytesIO(large_data)
    test_endpoint("5. Image exceeding upload-size limit (>10MB)", ("large.jpg", buf_large, "image/jpeg"))

if __name__ == "__main__":
    main()
