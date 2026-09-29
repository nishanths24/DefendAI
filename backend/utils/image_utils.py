import io
from PIL import Image

def load_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Invalid image format or corrupted file.") from e
