import pandas as pd
datagreen = pd.read_csv('housing-3.csv', delim_whitespace=True)

df = pd.read_csv("housing-3.csv")
print("✅ Columns in the CSV:")
print(df.columns.tolist())
