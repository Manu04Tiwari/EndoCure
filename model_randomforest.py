from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

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
