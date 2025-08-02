from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
# Load dataset
df = pd.read_excel('DataSet-fi.xlsx')

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns='Back pain')  # Assuming 'back pain' is target
y = df_encoded['Back pain']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=5)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)


# Calculate metrics
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"Recall: {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"AUC: {auc:.2f}")

#visualize

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No Back Pain', 'Back Pain'], filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title("Decision Tree Visualization")
plt.show()
