import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.utils import load_config, load_json
from src.metrics import mcnemar_test
from src.visualization import (
    plot_metrics_comparison,
    plot_mixed_class_analysis,
    plot_training_curves,
    plot_confusion_matrix_single,
)


EXPERIMENTS = ["baseline", "augmentation", "class_weighted"]


def compare_experiments(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Consolidates experimental outputs, performs statistical significance tests,
    generates comparison charts, and exports tables.
    """
    config = load_config(config_path)
    reports_dir = config["paths"]["reports_dir"]
    logs_dir = config["paths"]["logs_dir"]
    preds_dir = config["paths"]["predictions_dir"]
    cm_dir = config["paths"]["confusion_matrices_dir"]
    plots_dir = config["paths"]["plots_dir"]

    metrics_dict = {}
    histories_dict = {}
    preds_dict = {}

    for exp in EXPERIMENTS:
        m_path = os.path.join(reports_dir, f"{exp}_metrics.json")
        h_path = os.path.join(logs_dir, f"{exp}_history.json")
        p_path = os.path.join(preds_dir, f"{exp}_test_predictions.csv")

        if os.path.exists(m_path):
            metrics_dict[exp] = load_json(m_path)
        if os.path.exists(h_path):
            histories_dict[exp] = load_json(h_path)
        if os.path.exists(p_path):
            preds_dict[exp] = pd.read_csv(p_path)

    if not metrics_dict:
        print("No completed experiment metrics found to compare.")
        return pd.DataFrame()

    rows = []
    for exp in EXPERIMENTS:
        if exp not in metrics_dict:
            continue
        m = metrics_dict[exp]
        h = histories_dict.get(exp, {})

        training_time = h.get("total_training_time_seconds", 0.0)

        rows.append({
            "experiment": exp,
            "accuracy": round(m["accuracy"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "weighted_f1": round(m["weighted_f1"], 4),
            "mixed_precision": round(m["mixed_class"]["precision"], 4),
            "mixed_recall": round(m["mixed_class"]["recall"], 4),
            "mixed_f1": round(m["mixed_class"]["f1"], 4),
            "training_time_seconds": round(training_time, 1),
            "best_epoch": h.get("best_epoch", "N/A"),
        })

    comp_df = pd.DataFrame(rows)
    comp_csv_path = os.path.join(reports_dir, "experiment_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\nSaved consolidated experiment comparison to {comp_csv_path}")

    # Display comparison table
    print("\n" + "=" * 110)
    print("                      CONSOLIDATED TEST EXPERIMENTAL COMPARISON TABLE")
    print("=" * 110)
    print(comp_df.to_string(index=False))
    print("=" * 110)

    # 2. McNemar's Paired Tests
    mcnemar_results = {}
    if len(preds_dict) >= 2:
        print("\n--- McNemar's Paired Statistical Significance Tests ---")
        exp_names = list(preds_dict.keys())
        y_true = preds_dict[exp_names[0]]["ground_truth"].values

        for i in range(len(exp_names)):
            for j in range(i + 1, len(exp_names)):
                exp_a = exp_names[i]
                exp_b = exp_names[j]
                pred_a = preds_dict[exp_a]["predicted"].values
                pred_b = preds_dict[exp_b]["predicted"].values

                res = mcnemar_test(y_true, pred_a, pred_b)
                pair_key = f"{exp_a}_vs_{exp_b}"
                mcnemar_results[pair_key] = res
                sig_str = "Statistically Significant (p < 0.05)" if res["significant_at_05"] else "Not Significant (p >= 0.05)"
                print(f"Comparison: {exp_a.upper()} vs {exp_b.upper()} | Chi2: {res['statistic']:.4f} | p-value: {res['p_value']:.4e} | {sig_str}")

    # 3. Generate Comparative Plots
    if len(comp_df) > 0:
        plot_metrics_comparison(comp_df, save_path=os.path.join(plots_dir, "metrics_comparison.png"))
        plot_mixed_class_analysis(comp_df, save_path=os.path.join(plots_dir, "mixed_class_analysis.png"))

    if len(histories_dict) > 0:
        plot_training_curves(histories_dict, save_path=os.path.join(plots_dir, "training_curves.png"))

    # Plot individual confusion matrices
    for exp, m in metrics_dict.items():
        cm = m["confusion_matrix"]
        class_order = config["dataset"]["class_order"]
        plot_confusion_matrix_single(
            cm,
            class_names=class_order,
            experiment_name=exp,
            save_path=os.path.join(cm_dir, f"{exp}_confusion_matrix.png"),
        )
        plot_confusion_matrix_single(
            cm,
            class_names=class_order,
            experiment_name=exp,
            save_path=os.path.join(plots_dir, f"{exp}_confusion_matrix.png"),
        )

    # 4. Generate Comprehensive Experiment Report Markdown
    report_md_path = os.path.join(reports_dir, "experiment_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Research Report: Robust Deep Learning-Based Postharvest Fruit Quality Classification\n\n")
        f.write("## 1. Executive Summary & Comparative Results\n\n")
        f.write(comp_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. Statistical Significance Analysis (McNemar's Paired Test)\n\n")
        for k, v in mcnemar_results.items():
            f.write(f"- **{k.replace('_', ' ').title()}:**\n")
            f.write(f"  - Chi2 Statistic: `{v['statistic']:.4f}`, p-value: `{v['p_value']:.4e}`\n")
            f.write(f"  - Model A only correct: `{v['b_only_a_correct']}`, Model B only correct: `{v['c_only_b_correct']}`\n")
            f.write(f"  - Conclusion: **{'Significant difference at alpha=0.05' if v['significant_at_05'] else 'No statistically significant error difference'}**\n\n")

        f.write("## 3. Analysis of Minority Class (Mixed) Trade-Offs\n")
        f.write("Investigating whether data augmentation or class weighting successfully improved detection of the underrepresented Mixed class without catastrophic precision loss:\n\n")
        for exp in EXPERIMENTS:
            if exp in metrics_dict:
                m = metrics_dict[exp]
                mix = m["mixed_class"]
                ci = m.get("confidence_intervals_95", {}).get("mixed_recall", {})
                ci_str = f"[95% CI: {ci.get('ci_lower', 0):.4f} - {ci.get('ci_upper', 0):.4f}]" if ci else ""
                f.write(f"- **{exp.upper()}**:\n")
                f.write(f"  - Mixed Recall: `{mix['recall']*100:.2f}%` {ci_str}\n")
                f.write(f"  - Mixed Precision: `{mix['precision']:.4f}`\n")
                f.write(f"  - Mixed F1-Score: `{mix['f1']:.4f}`\n")
                f.write(f"  - Overall Accuracy: `{m['accuracy']*100:.2f}%` | Macro F1: `{m['macro_f1']:.4f}`\n\n")

    print(f"Generated comprehensive experiment report at {report_md_path}")
    return comp_df


if __name__ == "__main__":
    compare_experiments()
