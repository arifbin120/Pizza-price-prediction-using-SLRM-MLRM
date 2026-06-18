# 🍕 Pizza Price Prediction — End-to-End ML Pipeline

> Simple Linear Regression & Multiple Linear Regression on the Pizza V1 Dataset

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Results](#results)
- [Key Visualizations](#key-visualizations)
- [How to Run on Google Colab](#how-to-run-on-google-colab)

---

## 🔍 Project Overview

This project builds a complete, production-style machine learning pipeline to **predict pizza prices (in Indonesian Rupiah)** using both:

- **Simple Linear Regression** — one feature (strongest predictor)
- **Multiple Linear Regression** — all available features

The pipeline covers every stage from raw data ingestion to model evaluation, following best practices for preprocessing, encoding, scaling, and avoiding data leakage.

---

## 📦 Dataset

**File:** `pizza_v1.csv`  
**Rows:** 129 | **Columns:** 8

| Column | Type | Description |
|--------|------|-------------|
| `company` | Categorical | Pizza company (A–E) |
| `price_rupiah` | String → Float | Pizza price in Indonesian Rupiah (target variable) |
| `diameter` | Float | Pizza diameter in inches |
| `topping` | Categorical | Type of topping (chicken, pepperoni, mushrooms, etc.) |
| `variant` | Categorical | Pizza variant (double_signature, american_favorite, etc.) |
| `size` | Categorical | Size (small, medium, large, XL, jumbo, reguler) |
| `extra_sauce` | Binary | Whether extra sauce was added (yes/no) |
| `extra_cheese` | Binary | Whether extra cheese was added (yes/no) |

**Target Variable:** `price_rupiah`

---

## 🔧 Pipeline Steps

```
Raw CSV
   │
   ▼
1. Data Loading & Overview
   │  → shape, dtypes, null counts, duplicates
   ▼
2. Exploratory Data Analysis (EDA)
   │  → distributions, boxplots, scatter plots, price-by-category
   ▼
3. Data Preprocessing & Cleaning
   │  → parse price_rupiah string to float
   │  → median imputation (numerical), mode imputation (categorical)
   ▼
4. Label Encoding
   │  → LabelEncoder on all 6 categorical columns
   ▼
5. Correlation Analysis
   │  → heatmap, ranked correlation with target
   │  → auto-select best single feature for Simple LR
   ▼
6. Train-Test Split
   │  → 80% train / 20% test, random_state=42
   ▼
7. StandardScaler
   │  → fit on train only (no data leakage)
   │  → transform train & test
   ▼
8. Simple Linear Regression    9. Multiple Linear Regression
   │  → 1 feature                  │  → all features
   │  → regression line plot       │  → feature importance chart
   ▼                               ▼
10. Model Evaluation & Comparison
    → MAE, RMSE, R² for both models
```

---

## 📁 Project Structure

```
pizza-price-prediction/
│
├── pizza_v1.csv                        # Raw dataset
├── pizza_regression_pipeline.ipynb    # Main notebook (36 cells)
└── README.md                           # This file
```

---

## 🚀 Getting Started

### Option A — Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `pizza_regression_pipeline.ipynb`
3. In **Step 1**, uncomment the file-upload lines:
   ```python
   from google.colab import files
   uploaded = files.upload()   # choose pizza_v1.csv
   ```
4. Click **Runtime → Run all**

### Option B — Local Jupyter

```bash
# Clone or download the project folder
cd pizza-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook pizza_regression_pipeline.ipynb
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 📊 Results

| Model | MAE (Rp) | RMSE (Rp) | R² Score |
|-------|----------|-----------|----------|
| Simple Linear Regression | — | — | — |
| Multiple Linear Regression | — | — | — |

> **Note:** Fill in the metric values after running the notebook. Multiple LR is expected to outperform Simple LR due to using all available features.

### Interpretation

| Metric | Meaning |
|--------|---------|
| **MAE** | Average absolute error between predicted and actual price |
| **RMSE** | Penalises large errors more heavily than MAE |
| **R²** | Proportion of price variance explained by the model (1.0 = perfect) |

---

## 📈 Key Visualizations

The notebook generates the following plots:

1. **Categorical Distributions** — bar charts for company, topping, variant, size, extra_sauce, extra_cheese
2. **Diameter Analysis** — histogram + boxplot
3. **Price Distribution** — histogram, boxplot, scatter vs diameter
4. **Price by Category** — boxplots for all categorical features
5. **Correlation Heatmap** — ranked pairwise feature correlations
6. **Simple LR** — actual vs predicted, residual plot, regression line
7. **Multiple LR** — actual vs predicted, residual plot, feature importance bar chart
8. **Model Comparison** — side-by-side MAE / RMSE / R² bar chart

---

## 🧠 Key Concepts Used

| Concept | Purpose |
|---------|---------|
| **Label Encoding** | Convert categorical strings to integers for sklearn |
| **StandardScaler** | Normalise features to mean=0, std=1 for stable regression |
| **Train-Test Split** | Evaluate model on unseen data |
| **Data Leakage Prevention** | Scaler is fit only on training data, then applied to test |
| **Median Imputation** | Robust to outliers for numerical columns |
| **Mode Imputation** | Most frequent value for categorical columns |

---

## 👤 Author

Built as part of an end-to-end supervised learning exercise using the Pizza V1 dataset.

---

## 📄 License

This project is for educational purposes. Feel free to use and modify.
