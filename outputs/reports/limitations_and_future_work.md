# Limitations and Future Work

## 1. Experimental and Dataset Limitations

### 1.1 Dataset Imbalance and Representation
- **Class Skew:** The dataset exhibits a severe 10.86:1 majority-to-minority ratio (9,331 Good vs. 859 Mixed in the training set). While 107 test samples were sufficient for initial evaluation, a broader, multi-site postharvest benchmark with thousands of ambiguous/mixed samples is essential to rigorously test tail distributions.
- **Species Distribution Disparity:** Pomegranate constitutes 5,940 of the 11,664 Good samples (~51%), whereas other fruits (Apple, Banana, Guava, Lime/Lemon, Orange) have approximately ~1,100 samples per class. This creates a multi-fruit distribution skew across botanical categories.
- **Controlled Capture Conditions:** Most images in the dataset were acquired under controlled illumination and relatively clean backgrounds. Real-world postharvest conveyor belts, sorting facilities, and packing houses present motion blur, fruit rolling, partial occlusions, debris, and varying camera distances.

### 1.2 Performance Ceiling on Clean Splits
- On the untouched clean test set, pretrained DenseNet121 achieved >99.2% accuracy across all three methods. This creates a high performance ceiling on clean data where McNemar's paired test showed no statistically significant differences ($p > 0.05$).
- Consequently, clean test set accuracy alone is insufficient to evaluate postharvest robustness. True differentiation emerged only under environmental perturbation testing (e.g. Gaussian sensor noise and illumination shifts), where Baseline accuracy dropped to 74.71% while minority-augmented models sustained 90.73%.

### 1.3 Computational Budget and Single-Seed Policy
- Due to the large dataset size (19,526 images) and deep transfer learning architecture, models were trained under a fixed seed ($42$). While bootstrap resampling ($N=1000$) provided 95% confidence intervals, multi-seed repetitions across 5+ runs would further quantify run-to-run weight initialization variance.

---

## 2. Future Research Directions

### 2.1 Domain-Specific Self-Supervised Pretraining
- Transfer learning in this study utilized ImageNet-1K pretrained weights. ImageNet is primarily composed of everyday objects and animals. Future work should investigate self-supervised pretraining (e.g., DINOv2, SimCLR, SwAV) directly on large unlabeled agricultural and postharvest datasets.

### 2.2 Edge Hardware Optimization and Deployment
- Implement TensorRT, ONNX Runtime INT8/FP16 quantization, and pruning for deployment on resource-constrained embedded edge devices (e.g., NVIDIA Jetson Orin Nano, Raspberry Pi with Coral Edge TPU) suitable for real-time sorting lines (>60 FPS).

### 2.3 Multi-Modal Sensing (RGB + NIR / Hyperspectral)
- Surface RGB images cannot detect internal defects (e.g., core rot in apples, internal browning in bananas). Combining computer vision with near-infrared (NIR) or hyperspectral imaging would enable simultaneous internal and external quality grading.

### 2.4 Active Learning and Continuous Domain Adaptation
- Deploying fruit sorters across different harvest seasons and geographic regions leads to domain shift due to fruit variety morphology and natural weathering. Implementing continual test-time adaptation (TTA) and active learning will enable models to adapt to new orchards with minimal re-annotation.
