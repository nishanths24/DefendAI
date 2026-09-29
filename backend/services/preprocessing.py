from torchvision import transforms
from PIL import Image

def validate_image_size(file_size_bytes: int, max_size_mb: int) -> bool:
    return file_size_bytes <= max_size_mb * 1024 * 1024

def get_transform():
    # Common transformations for ResNet/CNNs
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image: Image.Image):
    transform = get_transform()
    return transform(image).unsqueeze(0)
