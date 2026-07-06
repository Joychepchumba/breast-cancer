# Breast Cancer Diagnosis Prediction — Project README

This README walks through **every cell** of `Breast_Cancer_ML_Project.ipynb` and explains what each line does and why it's there.

## Project Overview

**Goal:** Predict whether a breast tumor is **benign** or **malignant** using diagnostic measurements taken from digitized images of cell nuclei.

**Dataset:** The Breast Cancer Wisconsin Diagnostic dataset, loaded directly from `sklearn.datasets` (no external file needed). It contains 30 numeric features (e.g. radius, texture, perimeter, area, concavity) computed for each cell nucleus, plus a binary diagnosis label.

---

## Cell-by-Cell Breakdown

### 1. Import Libraries
```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.inspection import permutation_importance
```
- `warnings.filterwarnings('ignore')` silences non-critical warnings (e.g. convergence warnings) so the notebook output stays clean.
- `numpy` / `pandas` handle numerical arrays and tabular data.
- `matplotlib.pyplot` is used for all charts.
- `load_breast_cancer` pulls in the built-in dataset.
- `train_test_split` splits data into training/testing sets; `cross_val_score` runs k-fold cross-validation.
- `StandardScaler` standardizes features (mean 0, variance 1) — important for models like Logistic Regression that are sensitive to feature scale.
- `Pipeline` chains preprocessing and modeling steps together so they're applied consistently.
- `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `VotingClassifier` are the four models being compared (the last one is an ensemble of the first three).
- The `metrics` imports are the evaluation scores used later: accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix; `RocCurveDisplay` plots ROC curves directly from a fitted model.
- `permutation_importance` measures how much each feature contributes to the best model's performance.

### 2. Load Dataset From a Secondary Source
```python
cancer = load_breast_cancer(as_frame=True)
df = cancer.frame.copy()
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

print('Rows:', df.shape[0])
print('Features:', len(cancer.feature_names))
df.head()
```
- `load_breast_cancer(as_frame=True)` returns the dataset as a pandas-friendly bunch object, including a ready-made DataFrame (`cancer.frame`).
- `.copy()` creates an independent DataFrame `df` so the original object isn't mutated.
- A new `diagnosis` column translates the numeric `target` (0/1) into human-readable labels. **Note:** in this dataset, `0 = malignant` and `1 = benign` — the mapping in this cell matches that convention.
- The `print` statements report the dataset size (569 rows) and number of features (30).
- `df.head()` previews the first five rows.

### 3. Basic Data Understanding
```python
print(df['diagnosis'].value_counts())
print('\nMissing values:', df.isnull().sum().sum())
df.describe().T.head(10)
```
- `value_counts()` shows how many benign vs. malignant samples exist — useful for spotting class imbalance.
- `df.isnull().sum().sum()` adds up missing values across the entire DataFrame (this dataset is clean, so it should print 0).
- `df.describe().T.head(10)` transposes the summary statistics (count, mean, std, min, max, quartiles) so features become rows, and shows the first 10 features for a quick sanity check of scale and range.

### 4. Visualize Class Distribution
```python
counts = df['diagnosis'].value_counts().reindex(['Benign', 'Malignant'])
plt.figure(figsize=(6,4))
plt.bar(counts.index, counts.values)
plt.title('Diagnosis Distribution')
plt.ylabel('Number of Samples')
plt.show()
```
- Reorders the counts so bars appear consistently as Benign then Malignant.
- Draws a simple bar chart of how many samples fall into each class — a first check on class balance.

### 5. Visualize Key Feature Differences
```python
features = ['mean radius','mean texture','mean perimeter','mean area','mean concave points']
means = df.groupby('diagnosis')[features].mean().T

plt.figure(figsize=(8,5))
x = np.arange(len(features))
w = 0.38
plt.bar(x - w/2, means['Benign'], w, label='Benign')
plt.bar(x + w/2, means['Malignant'], w, label='Malignant')
plt.xticks(x, [f.replace('mean ', '') for f in features], rotation=25, ha='right')
plt.title('Average Cell Nucleus Features by Diagnosis')
plt.ylabel('Mean Value')
plt.legend()
plt.tight_layout()
plt.show()
```
- Picks five representative features and computes their **average value grouped by diagnosis**.
- Builds a grouped (side-by-side) bar chart comparing benign vs. malignant averages for each feature, revealing that malignant tumors tend to have larger/more irregular measurements.
- `x = np.arange(...)` creates evenly spaced bar positions; the `w` offset separates the Benign/Malignant bars so they don't overlap.

### 6. Prepare Train/Test Split
```python
X = df[cancer.feature_names]
y = df['target']  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Training samples:', X_train.shape[0])
print('Testing samples:', X_test.shape[0])
```
- `X` holds all 30 numeric features; `y` holds the binary target.
- `train_test_split` reserves 20% of the data for testing and 80% for training.
- `random_state=42` makes the split reproducible.
- `stratify=y` ensures the benign/malignant ratio is preserved in both the training and test sets.

### 7. Define Models
```python
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}

models['Soft Voting Ensemble'] = VotingClassifier(
    estimators=[
        ('lr', models['Logistic Regression']),
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting'])
    ],
    voting='soft'
)
```
- **Logistic Regression** is wrapped in a `Pipeline` with a `StandardScaler`, since logistic regression is sensitive to feature scale. `max_iter=5000` gives the solver enough iterations to converge.
- **Random Forest** uses 250 trees and `class_weight='balanced'` to automatically compensate for any class imbalance.
- **Gradient Boosting** uses scikit-learn's default boosting configuration.
- The **Soft Voting Ensemble** combines all three models, averaging their predicted probabilities (`voting='soft'`) rather than just majority-voting on hard labels — this typically gives smoother, more reliable predictions.

### 8. Train and Evaluate Models
```python
results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1-score': f1_score(y_test, predictions),
        'ROC-AUC': roc_auc_score(y_test, probabilities),
        'CV Accuracy Mean': cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    })

metrics = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
metrics
```
- Loops through each of the four models, fitting it on the training data and storing the fitted version in `fitted_models` (needed later for plotting/inspection).
- `predict()` gives hard class labels; `predict_proba(...)[:, 1]` gives the predicted probability of class `1` (benign), used for ROC-AUC.
- For each model, six metrics are computed:
  - **Accuracy** – overall proportion of correct predictions.
  - **Precision** – of predicted benign cases, how many were truly benign (minimizing false positives).
  - **Recall** – of actual benign cases, how many were correctly identified (minimizing false negatives — critical in medical contexts, since missing a malignant case is more dangerous than a false alarm).
  - **F1-score** – harmonic mean of precision and recall.
  - **ROC-AUC** – how well the model separates the two classes across all probability thresholds.
  - **CV Accuracy Mean** – average accuracy across 5-fold cross-validation on the *entire* dataset `X`/`y` (not just the train/test split), giving a more robust estimate of generalization.
- Results are collected into a DataFrame `metrics` and sorted so the best model (by ROC-AUC) appears first.

### 9. Visualize Model Performance
```python
plt.figure(figsize=(10,5))
x = np.arange(len(metrics))
width = 0.18
for i, col in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']):
    plt.bar(x + (i-2)*width, metrics[col], width, label=col)
plt.xticks(x, metrics['Model'], rotation=25, ha='right')
plt.ylim(0.85, 1.01)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend(ncols=3)
plt.tight_layout()
plt.show()
```
- Draws a grouped bar chart with one cluster of bars per model, and one bar per metric within each cluster.
- `(i-2)*width` offsets each metric's bars so all five sit neatly side-by-side without overlapping.
- The y-axis is zoomed into `0.85–1.01` since all models perform well and small differences would be invisible on a full 0–1 scale.

### 10. Confusion Matrix for the Best Model
```python
best_model_name = metrics.iloc[0]['Model']
best_model = fitted_models[best_model_name]

pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.title(f'Confusion Matrix: {best_model_name}')
plt.xticks([0,1], ['Malignant','Benign'])
plt.yticks([0,1], ['Malignant','Benign'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=16)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.show()
```
- Since `metrics` was sorted by ROC-AUC, `.iloc[0]` grabs the name of the top-performing model, and its fitted version is retrieved from `fitted_models`.
- `confusion_matrix` produces a 2×2 table of True Negatives / False Positives / False Negatives / True Positives.
- `plt.imshow(cm)` renders the matrix as a heatmap; the nested loop overlays the actual count values as text on each cell.
- Axis labels clarify which row/column corresponds to Malignant vs. Benign.

### 11. ROC Curve Comparison
```python
plt.figure(figsize=(7,5))
for name, model in fitted_models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=plt.gca())
plt.title('ROC Curves')
plt.show()
```
- Loops through all four fitted models and plots each one's ROC curve on the same axes (`plt.gca()` = "get current axes"), letting you visually compare how well each model trades off true positive rate vs. false positive rate.

### 12. Feature Importance (Permutation Importance)
```python
importance = permutation_importance(
    best_model, X_test, y_test, n_repeats=15, random_state=42, scoring='roc_auc'
)

imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance.importances_mean
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(8,5))
plt.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1])
plt.title(f'Top Predictive Features ({best_model_name})')
plt.xlabel('Permutation Importance')
plt.tight_layout()
plt.show()

imp_df
```
- **Permutation importance** measures how much the model's ROC-AUC score *drops* when a single feature's values are randomly shuffled — a bigger drop means that feature matters more. `n_repeats=15` repeats the shuffle 15 times per feature for a stable estimate.
- Results are packaged into `imp_df`, sorted so the most important features come first, and only the top 10 are kept.
- `plt.barh(...)` draws a horizontal bar chart; `[::-1]` reverses the order so the most important feature appears at the top of the chart instead of the bottom.

### Final Interpretation (Markdown Cell)
Summarizes that the model separates benign and malignant cases well on this dataset, but explicitly notes this is an **academic project only** — real clinical use would require external validation, medical review, privacy safeguards, and professional approval.

### 13. Save the Best Model
```python
import joblib
joblib.dump(best_model, 'breast_cancer_best_model.joblib')
print('Saved:', best_model_name)
```
- `joblib.dump` serializes the best-performing fitted model to disk as `breast_cancer_best_model.joblib`, so it can be reloaded later (e.g. `joblib.load(...)`) without retraining — useful for deployment or further testing.

---

## How to Run
1. Ensure `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `joblib` are installed.
2. Run all cells top to bottom in a Jupyter environment.
3. The notebook will print dataset info, show four charts (class distribution, feature comparison, model performance, confusion matrix, ROC curves, feature importance), print a metrics table, and save the best model to disk.

## Key Takeaways
- All four models achieve strong performance on this well-studied dataset, with the ensemble typically matching or slightly beating the individual models.
- Cross-validation accuracy is reported alongside the single train/test split to give a more trustworthy sense of how each model generalizes.
- Feature importance analysis shows which cell-nucleus measurements (e.g. size- and shape-related features like radius, perimeter, area, and concavity) most strongly drive the diagnosis prediction.
- This is a demonstration project — not a clinical tool.# breast-cancer
