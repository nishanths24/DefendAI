import json
import urllib.request
import urllib.error

mermaid_code = """
flowchart TD
    classDef entity fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef process fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#4a148c;
    classDef store fill:#eceff1,stroke:#607d8b,stroke-width:2px,color:#263238;
    classDef err fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#b71c1c;

    User([External Entity: USER])
    P1((1. Web Frontend<br/>Next.js + React))
    P2((2. FastAPI Prediction API<br/>/api/predict))
    P3((3. File Validation))
    P4((4. Input Integrity &<br/>Anomaly Detection))
    P5((5. Image Preprocessing))
    P6((6. ResNet18 Deep<br/>Learning Model))
    P7((7. Prediction Result))
    Dashboard([Frontend Result Dashboard])
    D1[(D1: Supabase Database<br/>OPTIONAL - Active if configured)]

    User -->|Upload image| P1
    P1 -->|HTTP multipart image request| P2
    P2 -->|Raw image bytes| P3
    
    P3 -.->|Invalid/corrupt| Err400[HTTP 400]
    P3 -.->|>10 MB| Err413[HTTP 413]
    Err400 -.->|Error JSON| P1
    Err413 -.->|Error JSON| P1

    P3 -->|Valid image| P4
    P4 -.->|Low-variance/anomalous| Err422[HTTP 422]
    Err422 -.->|Error JSON| P1

    P4 -->|Verified image| P5
    P5 -->|224x224, ImageNet norm| P6
    P6 -->|Class 0=FAKE, Class 1=REAL| P7
    
    P7 -->|Label, Confidence, Device| Dashboard
    Dashboard -->|Display Result| User

    P7 -->|Prediction logging<br/>filename, prediction, confidence| D1
    
    class User,Dashboard entity
    class P1,P2,P3,P4,P5,P6,P7 process
    class D1 store
    class Err400,Err413,Err422 err
"""

url = "https://kroki.io/"
data = json.dumps({
    "diagram_source": mermaid_code.strip(),
    "diagram_type": "mermaid",
    "output_format": "png"
}).encode('utf-8')

req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req) as response:
        with open("../docs/architecture/data-flow-diagram.png", "wb") as f:
            f.write(response.read())
    print("Downloaded to docs/architecture/data-flow-diagram.png")
except urllib.error.HTTPError as e:
    print(e.read().decode())
