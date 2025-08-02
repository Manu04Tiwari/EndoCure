from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
# Load the dataset 
df = pd.read_excel('DataSet-fi.xlsx')

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns='Back pain')  # Assuming 'Back-Pain' as the target
y = df_encoded['Back pain']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# ...existing code for training and predictions...

# Get predicted probabilities for ROC curve
y_prob_rf = model.predict_proba(X_test)[:, 1]

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob_rf):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()
