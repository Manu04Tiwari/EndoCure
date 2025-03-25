from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
df = pd.read_excel('DataSet-fi.xlsx')

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)


# Check the first few rows to make sure encoding worked correctly
print(df_encoded.head())

# Separate features (X) and target variable (y)
X = df_encoded.drop(columns='Back pain')  # Assuming Back-Pain as the target
y = df_encoded['Back pain']

# Initialize the scaler
scaler = StandardScaler()

# Scale the numerical features
X_scaled = scaler.fit_transform(X)

# Display the scaled features
print(X_scaled[:5])  # Displaying the first 5 rows of scaled data
