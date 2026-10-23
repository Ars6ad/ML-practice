# Dataset Audit Report: Postharvest Fruit Quality Dataset

## 1. Executive Summary
- **Total verified images on disk:** 19,526
- **Corrupted/Unreadable images:** 0
- **Exact duplicate image files (MD5):** 0
- **Official manifest entries:** 19,526
- **Training class imbalance ratio:** 10.86:1 (Good: 9,331 vs Mixed: 859)

## 2. Split and Class Distribution

| label   |   test |   train |   validation |   All |
|:--------|-------:|--------:|-------------:|------:|
| Bad     |    679 |    5430 |          679 |  6788 |
| Good    |   1167 |    9331 |         1166 | 11664 |
| Mixed   |    107 |     859 |          108 |  1074 |
| All     |   1953 |   15620 |         1953 | 19526 |

## 3. Data Integrity and Leakage Assessment
- **Image Format:** 100% 3-channel RGB JPEG images.
- **Integrity:** Zero corrupted images. All files opened and passed PIL integrity checks.
- **Leakage Guard:** Splits are strictly partitioned into train (80%), validation (10%), and test (10%). No image path appears in multiple splits.
- **Environmental & Fruit Varieties:** Dataset covers 6 distinct postharvest fruit varieties (Apple, Banana, Guava, Lime/Lemon, Orange, Pomegranate) across diverse natural lighting conditions.
