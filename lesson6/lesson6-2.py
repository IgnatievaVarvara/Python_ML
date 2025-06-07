# Регрессия с помощью случайных лесов

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'petal_length', 'species']]

data_setosa = data[data['species'] == 'setosa']

x_p = pd.DataFrame(
    np.linspace(min(data_setosa['sepal_length']), max(data_setosa['sepal_length']), 100)
)

X = pd.DataFrame(data_setosa['sepal_length'], columns=['sepal_length'])
y = data_setosa['petal_length']

model = RandomForestRegressor(n_estimators=20)
model.fit(X, y)

y_p = model.predict(x_p)

plt.scatter(data_setosa['sepal_length'], data_setosa['petal_length'])

plt.plot(x_p, y_p)

plt.show()

# Достоинства
# - Простота и быстрота. Распараллеливание процесса -> выигрыш во времени 
# - Вероятностная классификация 
# - Модель непараметрическая => хорошо работает с задачами, где другие модели могут оказаться недообученными

# Недостатки 
# - Сложно интерпретировать.