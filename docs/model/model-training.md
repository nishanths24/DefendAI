# Model Training

## Architecture
- **Base Model**: ResNet18
- **Input Size**: 224x224 pixels
- **Normalization**: ImageNet standards
- **Output Classes**: 2 (Binary Classification)
  - `0` = Fake
  - `1` = Real

## Final Weights
The final deployed model is stored at:
`backend/models/deepfake_model_diverse.pth`
