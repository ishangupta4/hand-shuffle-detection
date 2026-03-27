# Hand Shuffle AI -- Evaluation Report

*Generated: 2026-03-05 22:59*

## Overview

This report presents the Phase 7 evaluation results for the hand shuffle 
prediction model. The goal is to predict which hand holds a hidden object 
after a fast shuffle, using MediaPipe hand keypoints as input features.

- **Dataset**: 19 videos with engineered features (39 features per frame)
- **Evaluation**: Leave-One-Out Cross-Validation (LOOCV) -- 19 folds
- **Target**: `end_hand` (left=0, right=1)

## Model Comparison

| Model | Accuracy | Precision (L/R) | Recall (L/R) | F1 (L/R) | Macro F1 | ROC-AUC |
|-------|----------|-----------------|--------------|----------|----------|---------|
|  **cnn1d** | 0.842 | 0.82 / 0.88 | 0.90 / 0.78 | 0.86 / 0.82 | 0.840 | 0.967 |
| bilstm | 0.684 | 0.70 / 0.67 | 0.70 / 0.67 | 0.70 / 0.67 | 0.683 | 0.900 |
| transformer | 0.579 | 0.62 / 0.55 | 0.50 / 0.67 | 0.56 / 0.60 | 0.578 | 0.833 |

**Best model: cnn1d** (accuracy: 0.842)

## Confusion Matrix

![Confusion Matrix](outputs/cnn1d_confusion_matrix.png)

## ROC Curve

![ROC Curve](outputs/cnn1d_roc_curve.png)

## Per-Video Predictions

![Per-Video Accuracy](outputs/cnn1d_per_video_accuracy.png)

## Error Analysis

**3 misclassified out of 19 videos.**

| Video | True | Predicted | Confidence | Switched | Note |
|-------|------|-----------|------------|----------|------|
| 00003 | left | right | 0.505 | True | uncertain (close to 0.5) |
| 00008 | right | left | 0.580 | False | uncertain (close to 0.5) |
| 00013 | right | left | 0.560 | False | uncertain (close to 0.5) |

- Accuracy on **switched** videos: 0.917
- Accuracy on **non-switched** videos: 0.714
- Accuracy when end hand is **left**: 0.900
- Accuracy when end hand is **right**: 0.778
- Accuracy on **short** videos: 0.800
- Accuracy on **long** videos: 0.889

![Confidence Distribution](outputs/cnn1d_confidence_distribution.png)

## Feature Importance

### Random Forest Feature Importances

![RF Feature Importances](outputs/rf_feature_importances.png)

**Top 10 features:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | right_curl_velocity_index | 0.0558 |
| 2 | left_curl_velocity_index | 0.0540 |
| 3 | asymmetry_curl_middle | 0.0523 |
| 4 | right_curl_thumb | 0.0434 |
| 5 | left_fingertip_spread | 0.0404 |
| 6 | right_fingertip_spread | 0.0370 |
| 7 | right_curl_velocity_thumb | 0.0363 |
| 8 | left_curl_velocity_thumb | 0.0309 |
| 9 | right_wrist_velocity | 0.0306 |
| 10 | left_curl_index | 0.0303 |

### Permutation Importance

![Permutation Importance](outputs/permutation_importance.png)

## Temporal Analysis

## Conclusions

1. **Best model**: cnn1d with LOOCV accuracy of 0.842
2. **Assessment**: The model achieves above-chance performance, suggesting detectable patterns in hand pose during shuffling.
3. **Failure modes**: 0 confidently wrong predictions, 3 uncertain predictions near the decision boundary

### Recommended Next Steps

- Collect more videos (target 50+) to improve generalization
- Experiment with multi-task learning (predict both start and end hand)
- Try attention-based temporal pooling to focus on key moments
- Investigate whether video quality correlates with errors
- Consider ensemble methods combining the best DL and classical models