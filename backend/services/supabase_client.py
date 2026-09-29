import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY", "")

supabase: Client | None = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        supabase = None
else:
    print("Supabase credentials not found. Database integration disabled.")

def log_prediction(filename: str, prediction: str, confidence: float, user_id: str = None):
    if not supabase:
        return None
        
    try:
        data = {
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence
        }
        if user_id:
            data["user_id"] = user_id
            
        result = supabase.table("predictions").insert(data).execute()
        return result
    except Exception as e:
        print(f"Failed to log prediction to Supabase: {e}")
        return None
