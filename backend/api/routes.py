from fastapi import APIRouter, File, UploadFile, HTTPException
from core.config import settings
from utils.image_utils import load_image
from services.preprocessing import validate_image_size, preprocess_image
from services.anti_poisoning import run_anti_poisoning_checks
from services.prediction import model_service
from services.supabase_client import log_prediction

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@router.get("/ready")
def ready_check():
    return {"status": "ready"}

@router.post(
    "/api/predict",
    responses={
        422: {
            "description": "Input rejected by integrity/anomaly checks."
        },
        413: {
            "description": "File too large."
        },
        400: {
            "description": "Invalid image format or corrupted file."
        }
    }
)
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    file_bytes = await file.read()
    
    # 1. Validation
    if not validate_image_size(len(file_bytes), settings.MAX_UPLOAD_SIZE_MB):
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB.")
        
    try:
        image = load_image(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # 2. Anti-AI Poisoning Layer
    if not run_anti_poisoning_checks(image):
        raise HTTPException(status_code=422, detail="Image rejected by integrity checks.")
        
    # 3. Preprocessing
    tensor_image = preprocess_image(image)
    
    # 4. Inference
    result = model_service.predict(tensor_image)
    
    # 5. Log prediction to Supabase
    log_prediction(
        filename=file.filename,
        prediction=result["prediction"],
        confidence=result["confidence"]
    )
    
    return result
