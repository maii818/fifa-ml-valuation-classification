# FIFA Player Valuation & Performance Tier Classification — ML Project

An end-to-end machine learning system that predicts a football player's **market value** (regression) and classifies them into a **performance tier** (classification), built on a real FIFA player dataset. The project progresses from baseline models through hyperparameter tuning to a final **ensemble-based "Unified Scouting System."**

## Team

- Maii Walid
- Menna Emad
- Farah Ibrahim
- Farah Hossam
- Nourhan Essam

## 1. Dataset

**File:** `Fifa.csv` — 19,667 players × 9 columns

| Column | Description |
|---|---|
| Name, Country, Position, Team | Player identifiers / categorical attributes |
| Age | Player age |
| Overall_Rating | Current overall rating (used to derive the classification target) |
| Future Potential | Projected rating ceiling |
| Value Per M$ | Market value in millions — **regression target** |
| Total_Stats Score | Aggregate skill/stat score |

## 2. Exploratory Data Analysis

- No missing values or duplicate rows found in the raw data.
- **Value Per M$** is heavily right-skewed (skewness ≈ 7.98) — most players are low-value with a long tail of high-value outliers.
- **Correlation with market value:** Overall_Rating (0.56) > Future Potential (0.50) > Total_Stats Score (0.39) > Age (0.14).
- Outliers detected via IQR method across all numeric columns (most notably 2,390 outliers in `Value Per M$`).
- Average `Overall_Rating` varies by position (SW/RF highest, GK lowest).

## 3. Preprocessing & Feature Engineering

- **Train/test split:** 80/20, with stratified splits used for classification tasks.
- **Outlier handling:** IQR-based clipping, bounds computed on training data only and reused on test data (no leakage). Implemented as a reusable custom `OutlierClipper` transformer for the pipeline-based assignment.
- **Encoding:** One-Hot Encoding for `Position` and `Country` (fit on train only).
- **Scaling:** `StandardScaler` on numeric features (`Age`, `Future Potential`, `Total_Stats Score`).
- **Dropped columns:** `Name` and `Team` (unique identifiers, no predictive value).
- **Classification target:** `Overall_Rating` discretized into 4 tiers — **Low / Mid / High / Elite** — using the training set's 25th/50th/75th percentile thresholds (Q1=58, Q2=63, Q3=68), producing an approximately balanced class split (22%–28% per class).
- A production-style **`Pipeline` + `ColumnTransformer`** was built later in the project (imputation → outlier clipping → scaling / encoding) to guarantee zero data leakage across all models.

## 4. Regression Models (Predicting Market Value)

| Model | Test R² | Test RMSE | Notes |
|---|---|---|---|
| Baseline Linear Regression | 0.744 | 0.568 | |
| Polynomial Regression (degree 4) | 0.889 | 0.375 | Best-performing polynomial degree; train-test gap ≈ 0.0004 |
| Ridge (best α) | — | 0.375 | Outperformed Lasso — keeps all (correlated) OHE features |
| Lasso (best α) | — | 0.378 | Zeroed out 98 features (mostly polynomial terms) |
| KNN Regressor (tuned) | 0.803 | 3.14 | Overfit (train R² ≈ 1.0) before tuning correction |
| Random Forest Regressor (tuned) | 0.856 | 2.68 | Mean CV R² 0.83, std 0.043 — stable |
| SVR (RBF kernel, tuned) | 0.869 | 2.57 | Best single regression model; good generalization (gap 0.016) |
| **Voting Regressor Ensemble** (KNN+RF+SVR, weighted) | 0.868 | 2.57 | Comparable to best single model, more stable overall |

**Key finding:** Ridge regularization outperforms Lasso when many one-hot encoded features are present, since Ridge shrinks correlated features smoothly instead of discarding them.

## 5. Classification Models (Predicting Performance Tier)

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression (baseline) | 0.803 | Strong, balanced baseline |
| Logistic Regression (L1, tuned C) | 0.806 | Best regularization variant |
| GaussianNB | 0.719 | Best-suited Naive Bayes variant (continuous features) |
| BernoulliNB | 0.421 | Poor fit — designed for binary-only features |
| ComplementNB | 0.482 | Designed for text/count data, not ideal here |
| KNN Classifier (tuned) | 0.795 | Overfit before regularization (gap ≈ 0.21) |
| Random Forest Classifier (tuned) | 0.853 | Mean CV accuracy 0.846, std 0.0055 — very stable |
| SVM / SVC (RBF kernel, tuned) | 0.859 | Best single classifier; good generalization (gap 0.032) |
| **Voting Ensemble** (KNN+RF+SVM, soft voting) | 0.864 | |
| **Stacking Ensemble** (KNN+RF+SVM → Logistic Regression meta-model) | **0.868** | Best overall model, lowest CV std (0.0016) — most stable |

**Cross-validation:** Logistic Regression clearly outperformed and was more stable than Naive Bayes (std 0.0053 vs 0.0112) in Stratified K-Fold testing.

## 6. Ensemble Learning

Two ensemble strategies were implemented on top of the tuned KNN, Random Forest, and SVM/SVR models:

- **Voting Ensemble** — soft voting (classification) / weighted averaging (regression, weights `[1, 2, 2]` favoring RF and SVR) across the three base models.
- **Stacking Ensemble** — the three base models feed into a Logistic Regression meta-model trained via internal cross-validation (`cv=3`), letting the system learn which base model to trust for which player profile.

The **Stacking Classifier achieved the highest accuracy (86.8%) with the lowest cross-validation standard deviation (0.0016)** of any model tested, making it the final choice for the deployed classification component.

## 7. Final System: Unified Scouting System

A deployment function combines both pipelines into a single interface:

```python
def unified_scouting_system(player_data):
    predicted_value = voting_reg_pipeline.predict(player_df)[0]
    predicted_tier  = stacking_pipeline.predict(player_df)[0]
    return predicted_value, predicted_tier
```

Given a player's raw attributes (Age, Future Potential, Total_Stats Score, Country, Position), it returns both a predicted market value and a performance tier in one call — validated against random samples from the held-out test set.

## 8. Overall Results Summary

| Metric | Baseline (early assignment) | Final Advanced System | Improvement |
|---|---|---|---|
| Classification Accuracy | ~72% (Naive Bayes) | **86.8%** (Stacking Ensemble) | +14.8 pts |
| Regression R² | ~76% (Linear Regression) | **86.8%** (SVR / Voting Regressor) | +10.8 pts |
| Stability (CV std) | High variance | **0.0016** (Stacking) | Much more consistent |

## 9. Repository Contents

```
├── fifa_ml_project.ipynb        # Main notebook: EDA → preprocessing → all models → ensembles → deployment
├── Fifa.csv                     # Dataset
└── README.md
```

## 10. Tech Stack

Python · pandas · NumPy · scikit-learn (Linear/Polynomial/Ridge/Lasso, Logistic Regression, Naive Bayes, KNN, Random Forest, SVM/SVR, Voting/Stacking ensembles, Pipeline, ColumnTransformer, GridSearchCV) · Matplotlib · Seaborn

## Skills Demonstrated

Regression & Classification Modeling · Feature Engineering & Custom Transformers · Hyperparameter Tuning (GridSearchCV) · Cross-Validation & Stability Analysis · Overfitting/Underfitting Diagnosis · Ensemble Learning (Voting & Stacking) · Production-style ML Pipelines · Model Deployment
