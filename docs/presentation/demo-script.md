# Demo Script

1. **Introduction**: Introduce DefendAI as an application to detect synthetic image artifacts.
2. **Architecture Overview**: Briefly mention the Next.js frontend, FastAPI backend, and ResNet18 classifier.
3. **Happy Path**: 
   - Drag and drop a valid real image. Show the REAL verdict and confidence.
   - Upload a known deepfake image. Show the FAKE verdict.
4. **Security Protections**: 
   - Upload a corrupted image (Show HTTP 400 handling).
   - Upload a solid color / uniform image (Show HTTP 422 Anomaly Detection rejection).
   - Upload an oversized file > 10MB (Show HTTP 413 limit enforcement).
5. **Conclusion**: Highlight the held-out test accuracy of 88.10%.
