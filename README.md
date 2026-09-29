# DefendAI

DefendAI — Deepfake Detection and Anti-AI Poisoning Framework is a decoupled application that utilizes a deep learning classifier alongside an input validation layer to predict whether a given image is a genuine photograph (REAL) or AI-generated (FAKE).

## Features Actually Implemented
- Deepfake image classification (REAL vs. FAKE) using a custom-trained ResNet18 model.
- Drag-and-drop web frontend with immediate visual feedback, confidence scores, and processing device reporting.
- Input Integrity and Anomaly Detection layer to filter out pathological or zero-variance inputs before inference.
- Configurable robust REST API built on FastAPI.

*(Note: DefendAI currently supports static images only. It does not perform audio or video deepfake detection.)*

## Architecture
The application runs as a decoupled client-server architecture:
- **Frontend**: A Next.js/React UI deployed independently, which accepts user image uploads and renders prediction metrics.
- **Backend**: A FastAPI Python backend handling image validation, preprocessing, and inference via PyTorch. 

For full architectural documentation and visual diagrams, please refer to the `docs/architecture` directory.

## Technology Stack
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui
- **Backend API**: Python, FastAPI, Uvicorn
- **Machine Learning**: PyTorch, Torchvision (ResNet18)
- **Data Processing**: Pillow, OpenCV, NumPy

## Running Locally

### Backend Setup
1. Navigate to the `backend/` directory.
2. Ensure you have Python 3.12+ installed.
3. Install dependencies: `pip install -r requirements.txt`
4. Set up the `.env` file (see `.env.example`).
5. Run the server: `uvicorn main:app --reload`
   The backend will be available at `http://localhost:8000`.

### Frontend Setup
1. Navigate to the `frontend/` directory.
2. Install dependencies: `npm install`
3. Run the development server: `npm run dev`
   The frontend will be available at `http://localhost:3000`.

## API Endpoints
- `GET /ready`: Health check endpoint ensuring the API and model are loaded and functional.
- `POST /api/predict`: Core inference endpoint. Accepts `multipart/form-data` uploads with the key `file` and returns a JSON response containing `prediction`, `confidence`, and `device`.

## Model Information
- **Model**: PyTorch ResNet18
- **Normalization**: Standard ImageNet means and standard deviations.
- **Input Size**: 224x224 pixels
- **Output Classes**: `0 = FAKE`, `1 = REAL`
- **Location**: `backend/models/deepfake_model_diverse.pth`

*(Note: The deployed model utilizes the finalized DIVERSE dataset compilation. While DIVERSE V2 datasets exist in the project for future work, they are not the actively deployed model.)*

## Verified Test Metrics
Evaluated on a strictly held-out subset (2,000 images: 1,000 FAKE, 1,000 REAL).
- **Accuracy**: 88.10%
- **Precision**: 90.19%
- **Recall**: 85.50%
- **F1**: 87.78%
- **ROC-AUC**: 94.79%

## Security / Input-Integrity Behavior
DefendAI employs a lightweight pre-inference **Input Integrity and Anomaly Detection Layer**. This layer analyzes the variance of pixel values to validate input integrity.
- **Normal valid image**: Successfully processed (HTTP 200).
- **Corrupted image**: Caught by the file parser and rejected (HTTP 400).
- **Images > 10MB**: Caught by upload validation and rejected (HTTP 413).
- **Uniform / Low-Variance image**: Detected as anomalous by the integrity layer and rejected (HTTP 422).

*(Disclaimer: This layer is a heuristic check and is not a sophisticated adversarial defense mechanism. The model does not implement Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), or specific adversarial training.)*

## Limitations
1. Formal model evaluation is based strictly on a held-out evaluation subset. Performance generalization to unseen out-of-distribution AI generators or completely novel domains varies. It does not provide universal real-world accuracy guarantees.
2. The anomaly layer is a lightweight integrity heuristic, not a robust defense against mathematical adversarial attacks.
3. Supabase integration for logging predictions is implemented but remains strictly optional and disabled by default.
4. The system is restricted to classifying static images (no video or audio support).
