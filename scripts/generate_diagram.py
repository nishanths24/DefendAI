import base64
import zlib
import urllib.request
import os

def encode_kroki(text):
    compressed = zlib.compress(text.encode('utf-8'), 9)
    return base64.urlsafe_b64encode(compressed).decode('ascii')

mermaid_code = """
graph TD
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef frontend fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;
    classDef backend fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef ml fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef db fill:#eceff1,stroke:#607d8b,stroke-width:2px,color:#263238;
    classDef err fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#b71c1c;

    User([User]) -->|Upload Image| WebFrontend

    subgraph Frontend [Frontend System]
        WebFrontend["Web Frontend<br/>(Next.js + React + TypeScript)"]
        Dashboard["Result Dashboard<br/>(REAL/FAKE & Confidence)"]
    end

    WebFrontend -->|multipart/form-data| FastAPIBackend

    subgraph Backend [Backend API]
        FastAPIBackend["FastAPI Backend"]
        
        FileVal["File Validation<br/>(JPG/PNG/WebP, Max 10 MB, Corrupt-file detection)"]
        
        Security["Input Integrity & Anomaly Detection<br/>(Low-variance image detection)"]
        
        Preprocess["Image Preprocessing<br/>(Resize to 224x224, ImageNet normalization)"]
        
        Response["FastAPI Response (JSON)"]
    end

    FastAPIBackend --> FileVal
    FileVal -->|Valid| Security
    Security -->|Passed checks| Preprocess

    FileVal -.->|Invalid/Corrupt| Err400["HTTP 400 (Bad Request)"]
    FileVal -.->|Oversized| Err413["HTTP 413 (Payload Too Large)"]
    Security -.->|Anomalous| Err422["HTTP 422 (Unprocessable Entity)"]

    subgraph ML [Machine Learning Model]
        ResNet["ResNet18 Deep Learning Model<br/>(PyTorch, CUDA when available, Class 0 = FAKE, Class 1 = REAL)"]
        Pred["Prediction Result<br/>(REAL/FAKE, Confidence Score, Processing Device)"]
    end

    Preprocess --> ResNet
    ResNet --> Pred
    Pred --> Response

    subgraph Database [Database Logging Optional]
        Supabase[("Supabase Logging<br/>(Filename, Prediction, Confidence if configured)")]
    end

    Pred -->|Log details| Supabase

    Response --> Dashboard
    Err400 -.-> Dashboard
    Err413 -.-> Dashboard
    Err422 -.-> Dashboard

    class User user
    class WebFrontend,Dashboard frontend
    class FastAPIBackend,FileVal,Security,Preprocess,Response backend
    class Err400,Err413,Err422 err
    class ResNet,Pred ml
    class Supabase db
"""

import json

url = "https://kroki.io/"
data = json.dumps({
    "diagram_source": mermaid_code.strip(),
    "diagram_type": "mermaid",
    "output_format": "png"
}).encode('utf-8')

req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})
import urllib.error
try:
    with urllib.request.urlopen(req) as response:
        with open("../docs/architecture/system-architecture.png", "wb") as f:
            f.write(response.read())
    print("Downloaded to docs/architecture/system-architecture.png")
except urllib.error.HTTPError as e:
    print(e.read().decode())
