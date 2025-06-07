# Переобучение присуще всем деревьям принятия решений

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset('iris')

print(iris.head())

#sns.pairplot(iris, hue='species')

species_int = []
for r in iris.values:
    match r[4]:
        case 'setosa':
            species_int.append(1)
        case 'versicolor':
            species_int.append(2)
        case 'virginica':
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)
# print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int

data_versicolor = data[data['species'] == 2]
data_virginica = data[data['species'] == 3]

print(data_virginica.shape)
print(data_versicolor.shape)

data_versicolor_A = data_versicolor.iloc[:25, :]
data_versicolor_B = data_versicolor.iloc[25:, :]

data_virginica_A = data_virginica.iloc[:25, :]
data_virginica_B = data_virginica.iloc[25:, :]

data_df_A = pd.concat([data_virginica_A, data_versicolor_A], ignore_index=True)
data_df_B = pd.concat([data_virginica_B, data_versicolor_B], ignore_index=True)

x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']), 100)
x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns = ['sepal_length', 'petal_length']
)

fig, ax = plt.subplots(2, 4, sharex='col', sharey='row')

max_depth = [1, 3, 5, 7]

X = data_df_A[['sepal_length', 'petal_length']]
y = data_df_A['species']

j = 0
for md in max_depth:
    
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)

    ax[0, j].scatter(data_virginica_A['sepal_length'], data_virginica_A['petal_length'])
    ax[0, j].scatter(data_versicolor_A['sepal_length'], data_versicolor_A['petal_length'])

    y_p = model.predict(X_p)

    ax[0, j].contourf(
        X1_p,
        X2_p,
        y_p.reshape(X1_p.shape), 
        alpha = 0.4, 
        levels = 2, 
        cmap = 'rainbow', 
        zorder = 1
    )
    j += 1
    
    
X = data_df_B[['sepal_length', 'petal_length']]
y = data_df_B['species']

j = 0
for md in max_depth:
    
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X, y)

    ax[1, j].scatter(data_virginica_B['sepal_length'], data_virginica_B['petal_length'])
    ax[1, j].scatter(data_versicolor_B['sepal_length'], data_versicolor_B['petal_length'])

    y_p = model.predict(X_p)

    ax[1, j].contourf(
        X1_p,
        X2_p,
        y_p.reshape(X1_p.shape), 
        alpha = 0.4, 
        levels = 2, 
        cmap = 'rainbow', 
        zorder = 1
    )
    j += 1


plt.show()