from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
pd.__version__

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
# np.log(population)

print(type(cities['Population'][1]))
print(cities['Population'][1])

# print(type(cities[0:2]))
# cities[0:2]

# california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
# print(california_housing_dataframe.describe())
# print(california_housing_dataframe.head())
# california_housing_dataframe.hist('housing_median_age')


# print('test panda')
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
print(cities)

print(cities.index)
cities.reindex([0, 2, 1])
print(cities.index)

cities.reindex(np.random.permutation(cities.index))
print(cities)

cities.reindex([0, 4, 5, 2])
print(cities)