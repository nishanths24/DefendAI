"""
Input Integrity and Anomaly Detection Layer

Purpose:
Provides basic input validation by checking for anomalies in the image data.
This layer is intended to detect malformed, completely uniform, or highly 
anomalous inputs (like images with almost zero variance).

What it detects:
- Completely uniform images (e.g., solid gray squares)
- Very low-variance images (e.g., almost uniform noise)

What it does NOT detect:
- It does NOT detect advanced adversarial attacks like FGSM or PGD.
- It is not a substitute for true adversarial robustness or adversarial training.
"""

import numpy as np
from PIL import Image
from core.config import settings

def run_anti_poisoning_checks(image: Image.Image) -> bool:
    """
    Run input integrity and anomaly detection checks.
    
    Returns False if the image is detected as anomalous based on
    variance thresholding. Otherwise returns True.
    """
    if not settings.ANTI_POISONING_ENABLED:
        return True
        
    try:
        # Basic Anomaly/Integrity Check: Check for excessive noise or artifacts
        img_array = np.array(image)
        std_dev = np.std(img_array)
        
        # If standard deviation is extremely low, it might be a uniform color image (anomalous)
        if std_dev < settings.ANOMALY_THRESHOLD:
            return False
            
        return True
    except Exception:
        return False
