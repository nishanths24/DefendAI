# Setup and Deployment

## Backend
1. Create and activate a Python virtual environment.
2. Install required dependencies.
3. Verify `backend/models/deepfake_model_diverse.pth` is present.
4. Run the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```

## Frontend
1. Navigate to the `frontend/` directory.
2. Install Node.js dependencies (`npm install`).
3. Run the development server:
   ```bash
   npm run dev
   ```
