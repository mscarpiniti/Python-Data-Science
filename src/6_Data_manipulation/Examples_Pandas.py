# -*- coding: utf-8 -*-
"""
Examples of using the Pandas library for data manipulation.

Created on Sat Jan  7 12:17:59 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import pandas as pd


# %% Pandas Series

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data

data.values

data.index

data[1]
data[1:3]


s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s['b']


d = {'b': 1, 'a': 0, 'c': 2}
pd.Series(d)

pd.Series(d, index=['b', 'c', 'd', 'a'])


s.loc['b']
s.iloc[1]



# %% Pandas DataFrame

d1 = pd.Series([1., 2., 3.], index=['a', 'b', 'c'])
d2 = pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])
d3 = pd.Series([1., 2., 4.], index=['a', 'b', 'd'])
d = {'one': d1, 'two': d2, 'three': d3}
df = pd.DataFrame(d)


data = np.random.rand(3, 2)
df2 = pd.DataFrame(data, columns=['One', 'Two'], index=['a', 'b', 'c'])


ind  = df2.index
cnam = df2.columns
data = df2.values

df['two']
df.two

df[1:3]
df['b':'c']


data = np.random.rand(3, 2)
df3 = pd.DataFrame(data)

df3[1]
df3[1:2]


df.loc[['a', 'b']]
df.iloc[[1, 2]]
df.iat[1, 2]

df.loc[:'b',:'two']
df.iloc[:3,:2]


df.iloc[0, 1] = 5
df



# %% Basic functionalities

long_series = pd.Series(np.random.randn(1000))

long_series.head()
long_series.tail(3)

index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=['A', 'B', 'C'])
df[:2]
df.columns = [x.lower() for x in df.columns]
df
    
    
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s.to_numpy()
np.asarray(s)
    

df = pd.DataFrame({
        'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
        'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
        'three': pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})

row = df.iloc[1]
column = df['two']

df.sub(row, axis='columns')
df.sub(column, axis='index')
    
df = pd.DataFrame({'col1': np.random.randn(3), 'col2': np.random.randn(3)}, index=['a', 'b', 'c'])
for col in df:
    print(col)  



# %% Reading and writing data

titanic = pd.read_csv("data/titanic.csv")
titanic = pd.read_excel('titanic.xlsx', sheet_name='passengers')

titanic.to_csv('titanic.csv', index=False)
titanic.to_excel('titanic.xlsx', sheet_name='passengers', index=False)



# %% Basic data manipulations

titanic = pd.read_csv("../Data/titanic.csv")
ages = titanic["Age"]
age_sex = titanic[["Age", "Sex"]]

above_35 = titanic[titanic["Age"] > 35]
class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]

adult_names = titanic.loc[titanic["Age"] > 35, "Name"]
some_passengers = titanic.iloc[9:25, 2:5]
titanic.iloc[0:3, 3] = "anonymous"

table = pd.read_csv("data/mydata.csv")
table["col4"] = table["col1"] * 1.882
table["col5"] = table["col2"] / table["col3"]

table_new = table.rename(columns={"col1": "BETR801", "col2": "FR04014", "col3": "Roma"})

titanic["Age"].mean()
titanic[["Age", "Fare"]].median()
titanic.agg({'Age': ['min', 'max', 'median', 'skew'], 'Fare': ['min', 'max', 'median', 'mean']})
	
titanic[["Sex", "Age"]].groupby("Sex").mean()
titanic["Pclass"].value_counts()

titanic[["Age", "Fare"]].describe()

titanic.sort_values(by="Age")
titanic.sort_values(by=['Pclass', 'Age'], ascending=False)
table_subset = table.sort_index().groupby(["location"])


no2_data = pd.read_csv("data/air_quality_no2_long.csv")
pm25_data = pd.read_csv("data/air_quality_pm25_long.csv")

air_quality = pd.concat([pm25_data, no2_data], axis=0)
air_quality = pd.concat([pm25_data, no2_data], axis=1)


titanic = pd.read_csv("data/titanic.csv")
titanic["Name"].str.lower()
titanic["Surname"] = titanic["Name"].str.split(",").str.get(0)


idx = titanic["Name"].str.contains("Countess")
passengers = titanic[idx]

titanic["Sex_short"] = titanic["Sex"].replace({"male": "M", "female": "F"})


# %% Plotting data

titanic['Age'].plot()
titanic['Fare'].plot(kind='line')
titanic['Pclass'].plot(kind='bar')
titanic['Survived'].plot(kind='hist')
titanic.plot(x='Pclass',y='Fare',kind='scatter')


# %% Handling missing data

v1 = np.array([1, None, 3, 4])
v2 = np.array([1, np.nan, 3, 4])
v3 = pd.Series([1, np.nan, 2, None])

data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data_new = data.fillna(0)

N = df.isnull().sum().sum()
print(N)


df_a = df.dropna()
df_a

df_b = df.dropna(axis='columns')
df_b

