import os
from dotenv import load_dotenv

backend_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "development")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    
    _model_path_env = os.getenv("MODEL_PATH")
    if not _model_path_env:
        raise ValueError("CRITICAL: MODEL_PATH environment variable is not set. Cannot fall back to a default model silently.")
    
    # Resolve relative paths relative to backend directory
    if not os.path.isabs(_model_path_env):
        MODEL_PATH: str = os.path.abspath(os.path.join(backend_dir, _model_path_env))
    else:
        MODEL_PATH: str = _model_path_env

    DEVICE: str = os.getenv("DEVICE", "cpu")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
    ALLOWED_EXTENSIONS: str = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp")
    ANTI_POISONING_ENABLED: bool = os.getenv("ANTI_POISONING_ENABLED", "true").lower() == "true"
    ANOMALY_THRESHOLD: float = float(os.getenv("ANOMALY_THRESHOLD", "1.0"))

settings = Settings()
