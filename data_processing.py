import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset (replace 'DataSet-fi.xlsx' with your actual file name)
df = pd.read_excel('DataSet-fi.xlsx')

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute missing values in numerical columns with the mean
num_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Impute missing values in categorical columns with the mode (most frequent value)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Verify that missing values have been imputed
print(df.info())

# Display the first few rows of the updated DataFrame
print(df.head())

