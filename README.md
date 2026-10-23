# Robust Deep Learning-Based Postharvest Fruit Quality Classification Under Class Imbalance and Environmental Variations

A research-grade deep learning project implementing transfer learning with **DenseNet121** to classify postharvest fruit images into three quality grades:
- **Good** (Pristine, prime commercial grade)
- **Bad** (Decayed, damaged, or unmarketable)
- **Mixed** (Underrepresented minority class with minor surface blemishes, partial defect, or transitional ripening)

---

## 1. Experimental Overview

This project systematically evaluates three isolated strategies for mitigating class imbalance on a 19,526-image multi-fruit dataset:
1. **Experiment 1 (Baseline):** Transfer-learned DenseNet121 without imbalance interventions.
2. **Experiment 2 (Augmentation & Sampling):** DenseNet121 with minority-focused conservative data augmentation and controlled `WeightedRandomSampler` oversampling for Mixed fruit.
3. **Experiment 3 (Class-Weighted Loss):** DenseNet121 trained on the natural training distribution with analytical balanced class weights ($w_c = N / (K \cdot n_c)$) in PyTorch `CrossEntropyLoss`.

---

## 2. Key Empirical Findings

### Clean Test Set Results ($N = 1,953$)

| Experiment | Accuracy | Balanced Acc | Macro F1 | Mixed Precision | Mixed Recall | Mixed F1 | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | **99.59%** | **99.77%** | **0.9956** | 0.9907 | **100.00%** | **0.9953** | ~30.0 mins |
| **Augmentation** | 99.44% | 99.58% | 0.9943 | 0.9907 | **100.00%** | **0.9953** | ~20.1 mins |
| **Class-Weighted** | 99.23% | 99.23% | 0.9864 | 0.9550 | 99.07% | 0.9725 | ~11.0 mins |

### Environmental Perturbation Stress Testing

Under synthetic packhouse sensor noise (Gaussian noise $\sigma=0.08$):
- **Baseline:** Accuracy collapsed to **74.71%**, Mixed recall dropped to **58.88%**.
- **Class-Weighted:** Accuracy dropped to **82.39%**, Mixed recall was **89.72%**.
- **Augmentation & Sampling:** Maintained **90.73% accuracy** and **91.59% Mixed recall** (+32.7% gain in Mixed recall over Baseline!).

---

## 3. Installation & Environment Setup

### Prerequisites
- Python 3.10 – 3.12
- NVIDIA GPU with CUDA support recommended (tested on RTX 4050 6GB)

### Setup Virtual Environment
```powershell
# Using uv (recommended for ultra-fast setup)
uv venv .venv --python 3.12
.\.venv\Scripts\activate
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
uv pip install tabulate
```

---

## 4. Execution Commands

### 1. Run Automated Unit Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests
```

### 2. Run Comprehensive Dataset Audit
```powershell
.\.venv\Scripts\python.exe scripts/audit_dataset.py
```

### 3. Run Individual Experiments
```powershell
# Experiment 1: Baseline
.\.venv\Scripts\python.exe main.py --experiment baseline

# Experiment 2: Minority Augmentation & Sampling
.\.venv\Scripts\python.exe main.py --experiment augmentation

# Experiment 3: Class-Weighted Loss
.\.venv\Scripts\python.exe main.py --experiment class_weighted
```

### 4. Run Complete End-to-End Workflow (Audit $\rightarrow$ Train $\rightarrow$ Evaluate $\rightarrow$ Robustness $\rightarrow$ Compare)
```powershell
.\.venv\Scripts\python.exe main.py --experiment all
```

### 5. Evaluate Existing Checkpoints Only
```powershell
.\.venv\Scripts\python.exe main.py --evaluate-only
```

### 6. Generate Comparative Tables and Plots
```powershell
.\.venv\Scripts\python.exe scripts/compare_results.py
```

---

## 5. Project Directory Structure

```
fruit-quality-classification/
├── configs/
│   └── config.yaml                     # Master hyperparameter configuration
├── src/
│   ├── __init__.py
│   ├── dataset.py                      # Dataset class & DataLoader factory with sampler
│   ├── transforms.py                   # Preprocessing & augmentation pipelines
│   ├── models.py                       # DenseNet121 transfer learning architecture
│   ├── train.py                        # Training engine with AMP, early stopping, balanced loss
│   ├── evaluate.py                     # Evaluation engine, bootstrap CIs, error analysis
│   ├── metrics.py                      # Metrics calculations & McNemar test
│   ├── visualization.py                # Publication-quality plotting
│   ├── utils.py                        # Seed, hardware detection, I/O
│   └── robustness.py                   # Environmental perturbation testing
├── scripts/
│   ├── audit_dataset.py                # Dataset audit and integrity verification
│   └── compare_results.py              # Consolidated metrics & McNemar comparisons
├── tests/
│   ├── test_dataset.py                 # Dataset & loader unit tests
│   ├── test_metrics.py                 # Metric computation & bootstrap tests
│   └── test_models.py                  # Model forward pass & parameter group tests
├── outputs/
│   ├── checkpoints/                    # Saved best model checkpoints (.pth)
│   ├── dataset_audit/                  # Summaries and distributions
│   ├── logs/                           # Training history JSON files
│   ├── predictions/                    # Per-image prediction CSVs
│   ├── confusion_matrices/             # Normalized confusion matrix plots
│   ├── plots/                          # Comparison, curve, and analysis charts
│   └── reports/                        # Audit, experiment, and limitations reports
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Ignored caches, checkpoints, and datasets
├── main.py                             # Unified CLI entry point
└── README.md                           # Documentation
```

---

## 6. Artifact Deliverables Summary

- **Best Model Checkpoints:**
  - `outputs/checkpoints/baseline_best.pth`
  - `outputs/checkpoints/augmentation_best.pth`
  - `outputs/checkpoints/class_weighted_best.pth`
- **Numerical Tables:**
  - `outputs/dataset_audit/dataset_summary.csv`
  - `outputs/dataset_audit/class_distribution.csv`
  - `outputs/reports/experiment_comparison.csv`
  - `outputs/reports/robustness_results.csv`
  - `outputs/predictions/*_test_predictions.csv`
- **Visualizations:**
  - `outputs/plots/class_distribution.png`
  - `outputs/plots/training_curves.png`
  - `outputs/plots/metrics_comparison.png`
  - `outputs/plots/mixed_class_analysis.png`
  - `outputs/plots/robustness_comparison.png`
  - `outputs/confusion_matrices/*_confusion_matrix.png`
- **Research Documentation:**
  - `outputs/reports/dataset_audit_report.md`
  - `outputs/reports/experiment_report.md`
  - `outputs/reports/limitations_and_future_work.md`
