# Ансамблевые методы. В основе идея объединения нескольких переобученных (!)моделей для уменьшения эффекта переобучения. Это называется баггинг (bagging)
# Баггинг усредняет результаты -> оптимальной классификации

# Ансамбль случайных деревьев называется случайным лесом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

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

data_setosa = data[data['species'] == 1]
data_versicolor = data[data['species'] == 2]
data_virginica = data[data['species'] == 3]

x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']), 100)
x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns = ['sepal_length', 'petal_length']
)

fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')

ax[0].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
ax[0].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
ax[0].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])

ax[1].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
ax[1].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
ax[1].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])

ax[2].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
ax[2].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
ax[2].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])

#max_depth = [1, 3, 5, 7]

md = 6

X = data[['sepal_length', 'petal_length']]
y = data['species']

model1 = DecisionTreeClassifier(max_depth=md)
model1.fit(X, y)

y_p1 = model1.predict(X_p)

ax[0].contourf(
    X1_p,
    X2_p,
    y_p1.reshape(X1_p.shape), 
    alpha = 0.4, 
    levels = 2, 
    cmap = 'rainbow', 
    zorder = 1
)

# Bagging

model2 = DecisionTreeClassifier(max_depth=md)
b = BaggingClassifier(model2, n_estimators=20, max_samples=0.8, random_state=1)
b.fit(X, y)

y_p2 = b.predict(X_p)

ax[1].contourf(
    X1_p,
    X2_p,
    y_p2.reshape(X1_p.shape), 
    alpha = 0.4, 
    levels = 2, 
    cmap = 'rainbow', 
    zorder = 1
)

# Random Forest

model3 = RandomForestClassifier(n_estimators=20, max_samples=0.8, random_state=1)
model3.fit(X, y)

y_p3 = model3.predict(X_p)

ax[2].contourf(
    X1_p,
    X2_p,
    y_p3.reshape(X1_p.shape), 
    alpha = 0.4, 
    levels = 2, 
    cmap = 'rainbow', 
    zorder = 1
)




plt.show()