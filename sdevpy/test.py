""" Just to test things """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlwings as xw
# import json


app = xw.App()
wb = app.books['Book1']
sheet = wb.sheets[0]
sheet['A1'].value = 'aaaa'


# df = pd.DataFrame({
#     'store': ['A', 'A', 'B', 'B', 'A', 'B'],
#     'product': ['apple', 'banana', 'apple', 'banana', 'apple', 'banana'],
#     'sales': [10, 15, 20, 18, 12, 16],
#     'cost': [1, 1.5, 2, 1.8, 1.2, 1.6],
# })

# print(df)

# m = df.groupby('store')[['sales', 'cost']].sum()
# print(m)
# # Output: store A: 123.33, store B: 180.0

# # Group by multiple columns
# x = df.groupby(['store', 'product'])[['sales', 'cost']].sum()
# print(x)
# x.to_csv("C:\\temp\\pivot.csv")

# p = df.pivot_table(values='sales', index='store', columns='product', aggfunc='sum')
# print(p)
