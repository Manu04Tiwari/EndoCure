import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (make_scorer, precision_score, recall_score, f1_score, 
                             accuracy_score, roc_auc_score, confusion_matrix, matthews_corrcoef)

# ===== Load dataset =====
data = pd.read_excel('DataSet-fi.xlsx')

# ===== Prepare features =====
# Select only numeric columns for X
X = data.drop('Back pain', axis=1).select_dtypes(include=[np.number])
y = data['Back pain']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== Train Decision Tree =====
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_scaled, y_train)

# ===== Define metrics =====
scoring = {
    'Recall': make_scorer(recall_score),
    'Specificity': make_scorer(recall_score, pos_label=0),
    'Precision': make_scorer(precision_score),
    'F1-score': make_scorer(f1_score),
    'Accuracy': make_scorer(accuracy_score),
    'AUC': make_scorer(roc_auc_score)
}

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_validate(decision_tree_model, X_train_scaled, y_train, cv=cv, scoring=scoring)

# Display mean and std for each metric
for metric in scoring.keys():
    mean_score = np.mean(cv_results[f'test_{metric}'])
    std_score = np.std(cv_results[f'test_{metric}'])
    print(f'{metric} - Mean: {mean_score:.4f}, Std: {std_score:.4f}')

# ===== Feature Importances =====
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': decision_tree_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 Most Important Features:")
print(features_df.head(5))

# ===== Create directory for images =====
os.makedirs('images_produced', exist_ok=True)

# ===== Plot Decision Tree =====
plt.figure(figsize=(15, 10))
plot_tree(
    decision_tree_model,
    filled=True,
    feature_names=X_train.columns,
    class_names=['No Back Pain', 'Back Pain'],
    rounded=True,
    fontsize=10,
    max_depth=3
)
plt.tight_layout()
plt.savefig(os.path.join('images_produced', 'Decision_Tree.png'))
plt.close()

# ===== Predict & Confusion Matrix =====
y_pred = decision_tree_model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
print("Matthews Correlation Coefficient:", mcc)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Back Pain', 'Back Pain'],
    yticklabels=['No Back Pain', 'Back Pain']
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(os.path.join('images_produced', 'Confusion_Matrix.png'))
plt.close()
