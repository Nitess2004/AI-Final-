import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt



plt.show()

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# 1. LOAD DATASET
# Ensure the file name matches your dataset
data = pd.read_csv('/Users/nitesskatuwal/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print("\n===== FIRST DATA INSPECTION =====")
print(data.head())
print("\nDataset Shape:", data.shape)
print("\nData Types Summary:\n", data.dtypes.value_counts())

print("\n===== MISSING VALUE CHECK =====")
print(data.isnull().sum().sum())

# Class distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x=data["Attrition"])
plt.title("Class Distribution (Attrition: 0=No, 1=Yes)")
plt.savefig("class_distribution.png")

# 2. DATA PREPROCESSING
# Drop constant and redundant columns that offer no predictive value
data.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Encode Target (Yes -> 1, No -> 0)
le = LabelEncoder()
data['Attrition'] = le.fit_transform(data['Attrition'])

# One-Hot Encoding for all categorical text features
categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

X = data_encoded.drop("Attrition", axis=1)
y = data_encoded["Attrition"]
print("\nEncoding Completed. Features Shape:", X.shape)

# 3. DATA SPLICING (70% Training, 15% Validation, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split Results:")
print(f"Training: {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test: {X_test.shape}")

# 4. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("StandardScaler applied to all features.")

# 5. BASELINE LOGISTIC REGRESSION
lr_baseline = LogisticRegression(max_iter=1000)
lr_baseline.fit(X_train_scaled, y_train)
y_pred_base = lr_baseline.predict(X_test_scaled)

print("\n===== BASELINE MODEL PERFORMANCE =====")
print(classification_report(y_test, y_pred_base))

# 6. CLASS-WEIGHTED LOGISTIC REGRESSION (Addressing Imbalance)
lr_weighted = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
lr_weighted.fit(X_train_scaled, y_train)
y_pred_weight = lr_weighted.predict(X_test_scaled)

print("\n===== CLASS WEIGHTED MODEL PERFORMANCE =====")
print(classification_report(y_test, y_pred_weight))  
# Visualization: Confusion Matrices Comparison
fig, ax = plt.subplots(1,2, figsize=(12,5))

sns.heatmap(confusion_matrix(y_test, y_pred_base),
            annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Baseline Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, y_pred_weight),
            annot=True, fmt="d", cmap="Reds", ax=ax[1])
ax[1].set_title("Class Weighted Confusion Matrix")
plt.savefig("confusion_matrices_comparison.png")

# ROC Curve comparison for Baseline and Weighted models
y_prob_base = lr_baseline.predict_proba(X_test_scaled)[:,1]
y_prob_weight = lr_weighted.predict_proba(X_test_scaled)[:,1]

fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_base)
fpr_w, tpr_w, _ = roc_curve(y_test, y_prob_weight)

auc_b = roc_auc_score(y_test, y_prob_base)
auc_w = roc_auc_score(y_test, y_prob_weight)

plt.figure(figsize=(7,5))
plt.plot(fpr_b, tpr_b, label=f'Baseline (AUC={auc_b:.3f})')
plt.plot(fpr_w, tpr_w, label=f'Class Weighted (AUC={auc_w:.3f})')
plt.plot([0,1],[0,1],'--', color='black')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Analysis")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve_comparison.png")

# Feature importance analysis using model coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr_weighted.coef_[0]
})
coef_df["Abs"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Abs", ascending=False)

print("\nTop 10 Influential Features for Attrition:")
print(coef_df.head(10))

plt.figure(figsize=(8,6))
plt.barh(coef_df["Feature"].head(10),
         coef_df["Coefficient"].head(10), color='teal')
plt.gca().invert_yaxis()
plt.title("Feature Impact on Attrition (Weighted Model)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")

# 7. HYPERPARAMETER TUNING USING GRIDSEARCHCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced']
}

grid_lr = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=0
)

grid_lr.fit(X_train_scaled, y_train)

print("\n===== GRIDSEARCHCV BEST HYPERPARAMETERS =====")
print(grid_lr.best_params_)
print(f"Best Validation AUC: {grid_lr.best_score_:.4f}")

y_pred_tuned = grid_lr.predict(X_test_scaled)
y_prob_tuned = grid_lr.predict_proba(X_test_scaled)[:,1]
auc_tuned = roc_auc_score(y_test, y_prob_tuned)

print("\n===== TUNED MODEL PERFORMANCE =====")
print(classification_report(y_test, y_pred_tuned))
print(f"Final Test AUC: {auc_tuned:.3f}")

# Final Confusion matrix for the Tuned model
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_tuned),
            annot=True, fmt='d', cmap='Greens')
plt.title("Tuned Model: Confusion Matrix")
plt.savefig("confusion_matrix_tuned.png")

# FINAL PERFORMANCE COMPARISON: Combined ROC Curve
plt.figure(figsize=(7,5))
plt.plot(fpr_b, tpr_b, label=f'Baseline (AUC={auc_b:.3f})')
plt.plot(fpr_w, tpr_w, label=f'Class Weighted (AUC={auc_w:.3f})')

fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_tuned)
plt.plot(fpr_t, tpr_t, label=f'Tuned (AUC={auc_tuned:.3f})', linestyle='--')

plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Model Comparison (ROC Curves)")
plt.legend()
plt.grid(True)
plt.savefig("final_model_comparison_roc.png")

print("\nAll visuals generated and saved as PNG files.")
plt.show()