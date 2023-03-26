import pandas as pd
import os
import numpy as np

# Create dataframe from dictionary
alpha = [1, 2, 3, 4]
beta = ['a', 'b', 'c', 'd']
gamma = ['aa', 'bb', 'cc', 'dd']
dic = {'Alpha': alpha, 'Beta': beta, 'Gamma': gamma}
df = pd.DataFrame(dic)
print(list(df))

# Show summary information
print(df.info())

# Add content to the dictionary
dic['Delta'] = ['d1', 'd2', 'd3', 'd4']
df = pd.DataFrame(dic)

# Show first few rows
print(df.head())

# Replace index by one of the columns
# df.set_index('Gamma', inplace=True)

# Output to csv
path = r'D:\sandbox\Outputs'
file = os.path.join(path, 'panda_test.csv')
df.to_csv(file, index=False, columns=['Beta', 'Gamma', 'Delta'])  # Remove index, specify columns

# Create a new dataframe based on specified indexes
new_df = df.iloc[[0, 2, 3]]
print(new_df)

# Add/Create columns by some operations on an existing column
df['Add'] = [0.1, 0.5, 1.2, 5.7]
df['Operation'] = np.ceil(df['Add'])
print(df)

# Replace values where a condition is FALSE!
df['Operation'].where(df['Operation'] < 5, 5, inplace=True)
print(df)

# Number of items per value on a column
print(df['Alpha'].value_counts())

# Number of items per value on a column
print(df['Alpha'].describe())

# Remove a column
# print(df)
df.drop(["Add"], axis=1, inplace=True)
# print(df)

# Sort according to one column
df.sort_values(by="Operation", ascending=False, inplace=True)
print(df)

# Copy
# print(df)
df2 = df.copy()
df2['NewCat'] = ['z', 'y', 'c', 'b']
# print(df)

# Series
hourly_traffic = [120, 123, 124, 119, 196, 121, 118, 117, 500, 132]
series = pd.Series(hourly_traffic)
percentile = 0.95
quantile = series.quantile(percentile)
print(quantile)
print(np.percentile(hourly_traffic, percentile * 100))
