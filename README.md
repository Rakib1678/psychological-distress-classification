# A Hybrid Machine Learning Approach to Self-Assess Psychological Distress Due to Educational Stress Among University Students

Machine Learning course project.

Multi-class classification of anxiety, stress and depression levels in university
students from survey responses, using a full preprocessing → EDA → modelling →
hyperparameter-tuning pipeline across 15 algorithms.

## Dataset

- 1,977 survey responses, 39 features, from 15 universities in Bangladesh
  (9 public, 6 private)
- Instruments: GAD-7 (anxiety), PSS-10 (stress), PHQ-9 (depression)
- After outlier and duplicate removal: 1,785 records
- Three target labels: `Anxiety_Label`, `Stress_Label`, `Depression_Label`

The raw data file (`Raw Data.csv.xlsx`) is not committed. Place it in `data/`
before running the notebooks.

## Pipeline

1. **Preprocessing** — column mapping to short identifiers, ordinal encoding of
   demographics (age, academic year, CGPA), IQR outlier removal, duplicate
   removal, missing-value check
2. **EDA** — demographic distributions, label distributions, cross-tabs by
   gender / academic year / department, correlation analysis
3. **Feature engineering** — one-hot encoding of categoricals, Z-score
   normalization before and after
4. **Class balancing** — SMOTE oversampling on the training split
5. **Dimensionality reduction** — PCA retaining 95% variance
6. **Modelling** — 15 algorithms trained separately for each of the three targets
7. **Tuning** — GridSearchCV (3-fold) on the top 3 models per target

Split: 80/20 stratified train/test.

## Algorithms compared

Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Machine,
XGBoost, Decision Tree, Naive Bayes, AdaBoost, Perceptron, Ridge, Bagging,
Extra Trees, LightGBM, CatBoost, Gradient Boosting.

## Results (depression target, illustrative)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.94 | 0.94 | 0.94 | 0.94 |
| Support Vector Machine | 0.86 | 0.86 | 0.86 | 0.86 |
| CatBoost | 0.78 | 0.78 | 0.78 | 0.78 |

After tuning, SVM (`C=10`, `kernel='linear'`) reached 0.97 accuracy on this
target. Full per-target results are written out as CSV by the notebook.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── main.ipynb      # MAIN — complete pipeline, all 15 models
│   └── EDA.ipynb      # earlier standalone EDA + preprocessing
├── archive/
│   └── EDA_v2.ipynb   # near-duplicate of the EDA notebook
└── data/
    └── (place Raw Data.csv.xlsx here)
```

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/main.ipynb
```

Or open `main.ipynb` in Google Colab and upload the dataset when prompted.

## Outputs generated

`updated_dataset.csv`, `{anxiety,stress,depression}_model_results.csv`,
`{anxiety,stress,depression}_tuned_results.csv`,
`comprehensive_analysis_summary.json`.
