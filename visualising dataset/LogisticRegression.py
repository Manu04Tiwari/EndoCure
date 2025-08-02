from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay


# Load dataset
df = pd.read_excel('DataSet-fi.xlsx')

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns='Back pain')  # Assuming 'back pain' is target
y = df_encoded['Back pain']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression classifier
clf = LogisticRegression(max_iter=200)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)


# ...existing code for data loading and preprocessing...

# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
y_prob_lr = logreg.predict_proba(X_test)[:, 1]

# Calculate metrics
recall = recall_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
accuracy = accuracy_score(y_test, y_pred_lr)
auc = roc_auc_score(y_test, y_prob_lr)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
specificity = tn / (tn + fp)

print(f"Recall: {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {auc:.2f}")

# visualize

# ...existing code...

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Plot Confusion Matrix
ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=['No Back Pain', 'Back Pain'], cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()