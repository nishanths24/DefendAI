# Model Evaluation

## Model Details
- **Architecture**: ResNet18
- **Framework**: PyTorch
- **Model File**: `backend/models/deepfake_model_diverse.pth`
- **Class Mapping**: `0 = FAKE`, `1 = REAL`

## Formal Evaluation Metrics
The model was evaluated against a held-out diverse test dataset consisting of 2,000 images (1,000 FAKE, 1,000 REAL). 

- **Accuracy**: 88.10%
- **Precision**: 90.19%
- **Recall**: 85.50%
- **F1-score**: 87.78%
- **ROC-AUC**: 94.79%
- **Average confidence**: 88.60%

*Disclaimer: These metrics correspond exclusively to the held-out test set distribution and should NOT be presented as universal real-world accuracy.*

## Prediction Statistics
- **Correct predictions**: 1,762
- **Incorrect predictions**: 238
- **Fake detection rate**: 90.70%
- **Real detection rate**: 85.50%

### Per-Class Results
- **FAKE**: Precision: 86.22%, Recall: 90.70%, F1: 88.40%
- **REAL**: Precision: 90.19%, Recall: 85.50%, F1: 87.78%

## Confusion Matrix
|               | Predicted FAKE | Predicted REAL |
|---------------|----------------|----------------|
| **Actual FAKE** | 907            | 93             |
| **Actual REAL** | 145            | 855            |

## Real-World Exploratory Evaluation
A distinct, small-scale manual test yielded the following results, which are strictly exploratory and NOT statistically representative:
- **Genuine Photographs (8 tested)**: 75% accuracy (6 correct, 2 incorrect), Average Confidence: 83.67%
- **AI-Generated Images (10 tested)**: 40% detection rate (4 correct, 6 incorrect), Average Confidence: 81.60%
