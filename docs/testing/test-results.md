# Testing and Evaluation Report

## 1. Testing Strategy
The testing strategy for DefendAI encompasses multiple facets of the system to ensure robustness, accuracy, and operational integrity. The methodology is strictly partitioned into formal machine learning model evaluation on a held-out dataset, real-world exploratory testing, security and input-integrity validation, and comprehensive application health checks. 

## 2. Model Evaluation
The core classification engine utilizes a PyTorch-based **ResNet18** model (`backend/models/deepfake_model_diverse.pth`). The model was rigorously evaluated against a held-out diverse test dataset comprising 2,000 images (1,000 FAKE, 1,000 REAL). 

**Verified Results:**
- **Accuracy**: 88.10%
- **Precision**: 90.19%
- **Recall**: 85.50%
- **F1-score**: 87.78%
- **ROC-AUC**: 94.79%
- **Average confidence**: 88.60%

*Important Note:* These metrics are derived from a specific held-out test set. They represent the model's performance on that specific distribution and should NOT be presented as a universal guarantee of real-world accuracy across all unseen generators or domains.

### Per-Class Results
- **FAKE**: Precision: 86.22% | Recall: 90.70% | F1: 88.40%
- **REAL**: Precision: 90.19% | Recall: 85.50% | F1: 87.78%

The model achieved 1,762 correct predictions and 238 incorrect predictions, with a Fake detection rate of 90.70% and a Real detection rate of 85.50%.

## 3. Confusion Matrix
The performance breakdown on the 2,000-image dataset is illustrated below:

|               | Predicted FAKE | Predicted REAL |
|---------------|----------------|----------------|
| **Actual FAKE** | 907            | 93             |
| **Actual REAL** | 145            | 855            |

## 4. Real-World Exploratory Evaluation
Independent of the formal 2,000-image dataset, a manual exploratory evaluation was conducted to gauge behavior on arbitrary external data. 

**Genuine Photographs (8 images tested):**
- Correctly classified as REAL: 6
- Incorrectly classified as FAKE: 2
- **Exploratory genuine-photo accuracy**: 75%
- Average confidence: 83.67%

**AI-Generated Images (10 images tested):**
- Correctly detected as FAKE: 4
- Incorrectly classified as REAL: 6
- **Exploratory AI-detection rate**: 40%
- Average confidence: 81.60%

*Important Note:* These results are based on small, manually collected exploratory samples and are strictly NOT statistically representative of general real-world performance. They are documented separately and must not be combined with the formal evaluation metrics.

## 5. Security and Input Integrity Tests
DefendAI implements a lightweight, configurable heuristic check known as the **Input Integrity and Anomaly Detection Layer**. This layer analyzes image variance to validate data integrity prior to inference. It is expressly not a defense against sophisticated adversarial attacks (such as FGSM or PGD), nor does it utilize adversarial training.

**Verified Tests:**
1. **Normal valid image**: Expected accepted (HTTP 200) -> **PASS**
2. **Solid grey / uniform image**: Expected rejected by integrity check (HTTP 422) -> **PASS**
3. **Low-variance image**: Expected rejected by integrity check (HTTP 422) -> **PASS**
4. **Corrupted / invalid image**: Expected rejected (HTTP 400) -> **PASS**
5. **Image larger than 10 MB**: Expected rejected (HTTP 413) -> **PASS**

## 6. Application Health Checks
The operational health of the application has been verified through the following checks:
- Frontend builds successfully.
- Next.js frontend is connected to FastAPI backend.
- `/ready` endpoint returns ready status.
- `/api/predict` accepts valid images.
- Final model path is correctly loaded.
- CUDA inference works when available.
- Class mapping is 0 = FAKE and 1 = REAL.
- Upload size and file-format validation work.
- Corrupted files are rejected.
- Integrity/anomaly rejection works.
- No silent model fallback remains.

*(Note: Automated Playwright browser testing was attempted but could not be completed because the Playwright driver download failed. Full automated browser testing is not claimed.)*

## 7. Limitations
- Real-world exploratory datasets used for ad-hoc evaluation are small.
- AI-generation detection performance varies significantly across unseen generators and novel domains.
- The input integrity layer acts as a heuristic data validation mechanism and is not a sophisticated adversarial-defense system.
- Supabase logging is strictly optional and remains disabled unless explicit credentials are configured in the environment.
- Full automated browser testing was not completed due to a Playwright driver download issue.

## 8. Summary
The DefendAI system demonstrates strong baseline performance on its formal evaluation dataset, achieving an 88.10% accuracy and 94.79% ROC-AUC. The implementation successfully integrates an Input Integrity and Anomaly Detection layer to filter malformed and zero-variance inputs, while maintaining strict file validation and decoupled operational health. However, as exploratory testing indicates, performance constraints exist when encountering highly novel out-of-distribution synthetic data, reaffirming the necessity of using the tool as an assistive diagnostic rather than an infallible oracle.
