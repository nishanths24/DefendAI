# Security and Integrity

## Input Integrity and Anomaly Detection Layer
DefendAI employs a pre-inference integrity layer to detect malformed or anomalous inputs based on pixel variance. 

It rejects:
- Completely uniform images (solid color blocks).
- Extremely low-variance images.

*Note: This is a basic data integrity check and anomaly detection layer. It does NOT implement FGSM, PGD, or comprehensive adversarial robustness.*

## Validation Constraints
- **File Types**: `.jpg`, `.jpeg`, `.png`, `.webp`
- **File Size**: 10 MB maximum
- **CORS**: Configured to restrict access properly.
- **Fail-safe**: No silent model fallbacks (system strictly enforces explicitly configured model paths).
