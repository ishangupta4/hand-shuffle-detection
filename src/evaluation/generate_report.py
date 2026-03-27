"""Step 33 -- Generate a comprehensive evaluation report in markdown.

Embeds all plots, metrics tables, and analysis findings into a single
outputs/evaluation_report.md file.
"""

import json
import os
from datetime import datetime

import numpy as np


def generate_report(
    all_loocv_results: dict,
    all_metrics: dict,
    error_analysis: dict,
    feature_importance: dict,
    temporal_results: dict,
    ablation_results: dict = None,
    permutation_results: dict = None,
    output_dir: str = "outputs",
) -> str:
    """Generate markdown evaluation report.

    Args:
        all_loocv_results: dict mapping model_name -> loocv results
        all_metrics: dict mapping model_name -> metrics dict
        error_analysis: error analysis dict for best model
        feature_importance: RF feature importance dict
        temporal_results: dict with hidden_state and probability analysis
        ablation_results: ablation study dict (optional)
        permutation_results: permutation importance dict (optional)
        output_dir: where to write the report

    Returns:
        Path to the generated report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.md")

    lines = []

    # ---- Header ----
    lines.append("# Hand Shuffle AI -- Evaluation Report")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("## Overview\n")
    lines.append("This report presents the Phase 7 evaluation results for the hand shuffle ")
    lines.append("prediction model. The goal is to predict which hand holds a hidden object ")
    lines.append("after a fast shuffle, using MediaPipe hand keypoints as input features.\n")
    lines.append(f"- **Dataset**: 19 videos with engineered features (39 features per frame)")
    lines.append(f"- **Evaluation**: Leave-One-Out Cross-Validation (LOOCV) -- 19 folds")
    lines.append(f"- **Target**: `end_hand` (left=0, right=1)\n")

    # ---- Model Comparison Table ----
    lines.append("## Model Comparison\n")
    lines.append("| Model | Accuracy | Precision (L/R) | Recall (L/R) | F1 (L/R) | Macro F1 | ROC-AUC |")
    lines.append("|-------|----------|-----------------|--------------|----------|----------|---------|")

    # Sort by accuracy descending
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
    best_model_name = sorted_models[0][0] if sorted_models else "unknown"

    for model_name, m in sorted_models:
        acc = m.get("accuracy", 0)
        pl = m.get("precision_left", 0)
        pr = m.get("precision_right", 0)
        rl = m.get("recall_left", 0)
        rr = m.get("recall_right", 0)
        fl = m.get("f1_left", 0)
        fr = m.get("f1_right", 0)
        mf1 = m.get("macro_f1", 0)
        auc = m.get("roc_auc", float("nan"))

        marker = " **" if model_name == best_model_name else ""
        end_marker = "**" if marker else ""
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        lines.append(
            f"| {marker}{model_name}{end_marker} | {acc:.3f} | {pl:.2f} / {pr:.2f} | "
            f"{rl:.2f} / {rr:.2f} | {fl:.2f} / {fr:.2f} | {mf1:.3f} | {auc_str} |"
        )

    lines.append(f"\n**Best model: {best_model_name}** "
                 f"(accuracy: {all_metrics[best_model_name]['accuracy']:.3f})\n")

    # ---- Confusion Matrix ----
    best_metrics = all_metrics.get(best_model_name, {})
    cm_path = best_metrics.get("confusion_matrix_path", "")
    if cm_path:
        lines.append("## Confusion Matrix\n")
        lines.append(f"![Confusion Matrix]({cm_path})\n")

    # ---- ROC Curve ----
    roc_path = best_metrics.get("roc_curve_path", "")
    if roc_path:
        lines.append("## ROC Curve\n")
        lines.append(f"![ROC Curve]({roc_path})\n")

    # ---- Per-Video Accuracy ----
    pv_path = best_metrics.get("per_video_path", "")
    if pv_path:
        lines.append("## Per-Video Predictions\n")
        lines.append(f"![Per-Video Accuracy]({pv_path})\n")

    # ---- Error Analysis ----
    lines.append("## Error Analysis\n")
    if error_analysis:
        n_err = error_analysis.get("n_errors", 0)
        n_tot = error_analysis.get("total_videos", 19)
        lines.append(f"**{n_err} misclassified out of {n_tot} videos.**\n")

        if error_analysis.get("error_details"):
            lines.append("| Video | True | Predicted | Confidence | Switched | Note |")
            lines.append("|-------|------|-----------|------------|----------|------|")
            for e in error_analysis["error_details"]:
                lines.append(
                    f"| {e['video_id']} | {e['true']} | {e['predicted']} | "
                    f"{e['confidence']:.3f} | {e['switched']} | {e.get('note', '')} |"
                )
            lines.append("")

        # Pattern findings
        if "accuracy_switched" in error_analysis:
            lines.append(f"- Accuracy on **switched** videos: {error_analysis['accuracy_switched']:.3f}")
        if "accuracy_nonswitched" in error_analysis:
            lines.append(f"- Accuracy on **non-switched** videos: {error_analysis['accuracy_nonswitched']:.3f}")
        if "accuracy_end_left" in error_analysis:
            lines.append(f"- Accuracy when end hand is **left**: {error_analysis['accuracy_end_left']:.3f}")
        if "accuracy_end_right" in error_analysis:
            lines.append(f"- Accuracy when end hand is **right**: {error_analysis['accuracy_end_right']:.3f}")
        if "accuracy_short_videos" in error_analysis:
            lines.append(f"- Accuracy on **short** videos: {error_analysis['accuracy_short_videos']:.3f}")
        if "accuracy_long_videos" in error_analysis:
            lines.append(f"- Accuracy on **long** videos: {error_analysis['accuracy_long_videos']:.3f}")
        lines.append("")

        conf_plot = error_analysis.get("confidence_plot_path", "")
        if conf_plot:
            lines.append(f"![Confidence Distribution]({conf_plot})\n")

    # ---- Feature Importance ----
    lines.append("## Feature Importance\n")

    if feature_importance:
        lines.append("### Random Forest Feature Importances\n")
        rf_plot = feature_importance.get("plot_path", "")
        if rf_plot:
            lines.append(f"![RF Feature Importances]({rf_plot})\n")

        ranked = feature_importance.get("ranked_features", [])[:10]
        if ranked:
            lines.append("**Top 10 features:**\n")
            lines.append("| Rank | Feature | Importance |")
            lines.append("|------|---------|------------|")
            for i, (name, score) in enumerate(ranked, 1):
                lines.append(f"| {i} | {name} | {score:.4f} |")
            lines.append("")

    if permutation_results:
        lines.append("### Permutation Importance\n")
        perm_plot = permutation_results.get("plot_path", "")
        if perm_plot:
            lines.append(f"![Permutation Importance]({perm_plot})\n")

    if ablation_results:
        lines.append("### Ablation Study (Feature Groups)\n")
        abl_plot = ablation_results.get("plot_path", "")
        if abl_plot:
            lines.append(f"![Ablation Study]({abl_plot})\n")

        lines.append(f"Baseline accuracy: {ablation_results.get('baseline_accuracy', 0):.3f}\n")
        lines.append("| Removed Group | Features Removed | Accuracy Without | Accuracy Drop |")
        lines.append("|---------------|-----------------|-----------------|---------------|")
        for a in ablation_results.get("ablations", []):
            lines.append(
                f"| {a['removed_group']} | {a['n_features_removed']} | "
                f"{a['accuracy_without']:.3f} | {a['accuracy_drop']:+.3f} |"
            )
        lines.append("")

    # ---- Temporal Analysis ----
    lines.append("## Temporal Analysis\n")
    if temporal_results:
        hidden_plot = temporal_results.get("hidden_state", {}).get("hidden_state_plot", "")
        if hidden_plot:
            lines.append("### Hidden State Trajectories\n")
            pca_var = temporal_results["hidden_state"].get("pca_variance_explained", [])
            if pca_var:
                lines.append(f"PCA explained variance: {pca_var[0]:.1%} (PC1), {pca_var[1]:.1%} (PC2)\n")
            lines.append(f"![Hidden State Trajectories]({hidden_plot})\n")

        prob_plot = temporal_results.get("probability", {}).get("probability_plot", "")
        if prob_plot:
            lines.append("### Frame-by-Frame Prediction Probability\n")
            lines.append("These plots show how the model's prediction evolves as it sees more frames. ")
            lines.append("A model that only uses the reveal would show a flat line until the end.\n")
            lines.append(f"![Temporal Probabilities]({prob_plot})\n")

    # ---- Conclusions ----
    lines.append("## Conclusions\n")

    best_acc = all_metrics[best_model_name]["accuracy"]
    if best_acc >= 0.7:
        verdict = "The model achieves above-chance performance, suggesting detectable patterns in hand pose during shuffling."
    elif best_acc > 0.55:
        verdict = "The model shows marginal performance above chance. The signal may be weak or the dataset too small."
    else:
        verdict = "Performance is near chance level. The hand shuffle pattern may not be detectable from keypoint data alone with this dataset size."

    lines.append(f"1. **Best model**: {best_model_name} with LOOCV accuracy of {best_acc:.3f}")
    lines.append(f"2. **Assessment**: {verdict}")

    # Failure modes
    if error_analysis and error_analysis.get("error_details"):
        n_confident = error_analysis.get("n_confident_errors", 0)
        n_uncertain = error_analysis.get("n_uncertain_errors", 0)
        lines.append(f"3. **Failure modes**: {n_confident} confidently wrong predictions, "
                     f"{n_uncertain} uncertain predictions near the decision boundary")

    lines.append("\n### Recommended Next Steps\n")
    lines.append("- Collect more videos (target 50+) to improve generalization")
    lines.append("- Experiment with multi-task learning (predict both start and end hand)")
    lines.append("- Try attention-based temporal pooling to focus on key moments")
    lines.append("- Investigate whether video quality correlates with errors")
    lines.append("- Consider ensemble methods combining the best DL and classical models")

    # Write
    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Report saved to {report_path}")
    return report_path