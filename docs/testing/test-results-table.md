# Testing and Evaluation Tables

## 1. Model Metrics (Held-Out Test Set: 2,000 Images)
| Metric | Value |
|--------|-------|
| Accuracy | 88.10% |
| Precision | 90.19% |
| Recall | 85.50% |
| F1-score | 87.78% |
| ROC-AUC | 94.79% |
| Average Confidence | 88.60% |
| Fake Detection Rate | 90.70% |
| Real Detection Rate | 85.50% |

## 2. Confusion Matrix
| | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actual FAKE** | 907 | 93 |
| **Actual REAL** | 145 | 855 |

## 3. Real-World Exploratory Testing (Manually Collected)
| Dataset Category | Samples Tested | Correct Predictions | Incorrect Predictions | Accuracy / Detection Rate | Average Confidence |
|---|---|---|---|---|---|
| Genuine Photographs | 8 | 6 (REAL) | 2 (FAKE) | 75% | 83.67% |
| AI-Generated Images | 10 | 4 (FAKE) | 6 (REAL) | 40% | 81.60% |

## 4. Security and Input-Integrity Testing
| Test ID | Input | Expected Behavior | HTTP Status | Result |
|---|---|---|---|---|
| 1 | Normal valid image | Accepted | 200 | PASS |
| 2 | Solid grey / uniform image | Rejected by integrity check | 422 | PASS |
| 3 | Low-variance image | Rejected by integrity check | 422 | PASS |
| 4 | Corrupted / invalid image | Rejected | 400 | PASS |
| 5 | Image larger than 10 MB | Rejected | 413 | PASS |

## 5. Application Health Checks
| Component / Functionality | Verification Status |
|---|---|
| Frontend builds successfully | PASS |
| Next.js frontend is connected to FastAPI backend | PASS |
| `/ready` endpoint returns ready status | PASS |
| `/api/predict` accepts valid images | PASS |
| Final model path is correctly loaded | PASS |
| CUDA inference works when available | PASS |
| Class mapping is 0 = FAKE and 1 = REAL | PASS |
| Upload size and file-format validation work | PASS |
| Corrupted files are rejected | PASS |
| Integrity/anomaly rejection works | PASS |
| No silent model fallback remains | PASS |
| Automated Playwright browser testing | FAILED (Driver download issue; incomplete) |
