import pandas as pd

# Load the dataset from Excel
df = pd.read_excel('DataSet-fi.xlsx')

# Show the first few rows of the dataset
print(df.head())

# Check for any missing values or other issues in the dataset
print(df.info())

