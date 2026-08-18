import pandas as pd

df = pd.read_csv('data/survey.csv')
print(df.shape)
print(df.info())
print(df.head())

# Look at annual_salary_usd
print(df['annual_salary_usd'].describe())
print(df['annual_salary_usd'].isna().sum())
# Visualize it - is it normal? skewed? any weird values?


