# Метод опорных векторов (SCM -- support vector machine) -- классиффикация и регрессии
# Разделяющая классификаия
#Выбирается линия с максимальным отступом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset("iris")

print(iris.head())

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

plt.show()

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = SVC(kernel = 'linear', C=10000)
model.fit(X, Y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=400,
            facecolor = 'none',
            edgecolors = 'black',
            )
plt.show()

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = SVC(kernel = 'linear', C=10000)
model.fit(X, Y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1],
            s=400,
            facecolor = 'none',
            edgecolors = 'black',
            )

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns = ['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)

X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha = 0.4)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha = 0.4)

plt.show()

# ДЗ. Убрать из данных iris часть точек (на которых обучаемся) и убедиться, что на предсказание влияют только опорные вектора

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset("iris")
data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

# Удаляем часть точек (оставляем только 30% данных)
np.random.seed(42)
reduced_data = data_df.sample(frac=0.3)

X = reduced_data[['sepal_length', 'petal_length']]
Y = reduced_data['species']

data_df_seposa = reduced_data[reduced_data['species'] == 'setosa']
data_df_versicolor = reduced_data[reduced_data['species'] == 'versicolor']

plt.figure(figsize=(10, 6))
plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'], label='setosa')
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'], label='versicolor')

model = SVC(kernel='linear', C=10000)
model.fit(X, Y)

print("Опорные векторы:")
print(model.support_vectors_)

plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=200,
            facecolor='none',
            edgecolors='black',
            linewidths=2,
            label='Опорные векторы')

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
    columns=['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)
X_p['species'] = y_p

X_p_setosa = X_p[X_p['species'] == 'setosa']
X_p_versicolor = X_p[X_p['species'] == 'versicolor']

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'],
            alpha=0.1, color='blue', marker='s')
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'],
            alpha=0.1, color='orange', marker='s')

# Создаем модель только на опорных векторах
model_sv_only = SVC(kernel='linear', C=10000)
model_sv_only.fit(model.support_vectors_, Y.iloc[model.support_])

# Сравниваем предсказания полной модели и модели только на опорных векторах
comparison = model.predict(X_p.drop('species', axis=1)) == model_sv_only.predict(X_p.drop('species', axis=1))
print("Процент совпадения предсказаний:", np.mean(comparison)*100)

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

plt.scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

plt.show()

# В случае, если данные перекрываются, то идеальной границы не существует. У модели существует гиперпараметр, который опрделеяет "размытие" отсутпа.

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

c_value = [[10000, 1000, 100, 10], [1., 0.1, 0.01, 0.001]]

fig, ax = plt.subplots(2, 4, sharex = 'col', sharey = 'row')

for i in range(2):
  for j in range(4):

      ax[i, j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
      ax[i, j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

      # Если  С большое, то отступ задается "жестко". Чем меньше С, тем отступ становится более "размытым"

      model = SVC(kernel = 'linear', C = c_value[i][j])
      model.fit(X, Y)

      # print(model.support_vectors_)

      ax[i, j].scatter(model.support_vectors_[:,0],
                  model.support_vectors_[:,1],
                  s=400,
                  facecolor = 'none',
                  edgecolors = 'black',
                  )
      x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
      x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

      X1_p, X2_p = np.meshgrid(x1_p, x2_p)

      X_p = pd.DataFrame(
          np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns = ['sepal_length', 'petal_length']
      )

      y_p = model.predict(X_p)

      X_p['species'] = y_p

      X_p_virginica = X_p[X_p['species'] == 'virginica']
      X_p_versicolor = X_p[X_p['species'] == 'versicolor']

      ax[i, j].scatter(X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha = 0.1)
      ax[i, j].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha = 0.1)
plt.show()

# Достоинства
# - Зависимость от небольшого числа опорных векторов => компактность модели
# - После обучения предсказания проходят очень быстро
# - На работу метода влияют ТОЛЬКО точки, находящиесявозле отступов, поэтому методы подходят для многомерных данных

# Недостатки
# - При большом количетсве обучающих образов могут быть значительные вычислительные затраты
# - Большая зависимость от размытости С. Поиск может привести к большим вычислительным затаратам
# - У результатов отстутствует вероятностная интерпретация

# Деревья решений и случайные леса
# СЛ -- непараметрический алгоритм
# СЛ -- пример ансамблевого метода. онованного на агрегации результатов множества простых моделей
# В реализации дерева принятия решений в машинном обучении, вопросы обычно ведут к разделению данных по осям, т.е. каждый узел разбивает данные на две группы по одному из признаков

from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset("iris")

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
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int_df

print(data)

data_df = data[(data['species'] == 1) | (data['species'] == 2)]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 1]
data_df_versicolor = data_df[data_df['species'] == 2]

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = DecisionTreeClassifier()
model.fit(X, Y)

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns = ['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)

# plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha = 0.4)
# plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha = 0.4)

plt.contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha = 0.4, levels = 2, cmap = 'rainbow', zorder = 1)
plt.show()

# Деревья решений и случайные леса
# СЛ -- непараметрический алгоритм
# СЛ -- пример ансамблевого метода. онованного на агрегации результатов множества простых моделей
# В реализации дерева принятия решений в машинном обучении, вопросы обычно ведут к разделению данных по осям, т.е. каждый узел разбивает данные на две группы по одному из признаков

from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset("iris")

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
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int_df

print(data)

data_df = data[(data['species'] == 3) | (data['species'] == 2)]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_seposa = data_df[data_df['species'] == 3]
data_df_versicolor = data_df[data_df['species'] == 2]

plt.scatter(data_df_seposa['sepal_length'], data_df_seposa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = DecisionTreeClassifier()
model.fit(X, Y)

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns = ['sepal_length', 'petal_length']
)

y_p = model.predict(X_p)

# plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha = 0.4)
# plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha = 0.4)

plt.contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha = 0.4, levels = 2, cmap = 'rainbow', zorder = 1)
plt.show()

# меняем параметр max_depth

from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset("iris")

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
print(species_int_df.head())

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int_df

print(data)

data_df = data[(data['species'] == 3) | (data['species'] == 2)]

X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 3]
data_df_versicolor = data_df[data_df['species'] == 2]

max_depth = [[1, 2, 3, 4], [5, 6, 7, 8]]

fig, ax = plt.subplots(2, 4, sharex = 'col', sharey = 'row')

for i in range(2):
  for j in range(4):
    ax[i, j].scatter(data_df_virginica['sepal_length'], data_df_virginica['petal_length'])
    ax[i, j].scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

    model = DecisionTreeClassifier(max_depth = max_depth[i][j])
    model.fit(X, Y)

    x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
    x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

    X1_p, X2_p = np.meshgrid(x1_p, x2_p)

    X_p = pd.DataFrame(
          np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns = ['sepal_length', 'petal_length']
      )

    y_p = model.predict(X_p)

    ax[i, j].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha = 0.4, levels = 2, cmap = 'rainbow', zorder = 1)

plt.show()