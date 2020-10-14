import pandas
import os

print(os.getcwd())
os.chdir('stock-market-dataset')
print(os.getcwd())

data = pandas.read_csv('FirstTest.csv')

print(data)

