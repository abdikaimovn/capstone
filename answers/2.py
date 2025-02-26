import numpy as np
import pandas as pd

# Загрузка данных
df = pd.read_csv("q2.csv")  # Укажи правильное название файла

# Преобразование данных в числовой формат
df = df.apply(pd.to_numeric, errors='coerce')

# Разделяем X и Y
X = df.iloc[:, :-1].values  # Все столбцы, кроме последнего (признаки)
Y = df.iloc[:, -1].values.reshape(-1, 1)  # Последний столбец (целевая переменная)

# Нормализация данных (Z-score)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std  # Приведение данных к стандартному виду

# Добавляем столбец единиц для θ0
X = np.c_[np.ones(X.shape[0]), X]  # Добавляем столбец 1 в X

# Градиентный спуск
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X @ theta
    errors = predictions - Y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient_descent(X, Y, theta, alpha, iterations):
    m = len(Y)
    cost_history = []

    for i in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - Y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, Y, theta))

    return theta, cost_history

# Параметры градиентного спуска
alpha = 0.1
iterations_list = [10, 100, 1000]

# Запускаем градиентный спуск для разных итераций
for iterations in iterations_list:
    theta = np.zeros((X.shape[1], 1))  # Инициализация нулями
    theta, cost_history = gradient_descent(X, Y, theta, alpha, iterations)

    print(f"\n# Итераций: {iterations}")
    print(f"Функция стоимости (округлённая): {round(cost_history[-1])}")
    print(f"Максимальное значение θ (округлённое): {round(np.max(theta))}")
import numpy as np
import pandas as pd

# Загрузка данных
df = pd.read_csv("q2.csv")  # Укажи правильное название файла

# Преобразование данных в числовой формат
df = df.apply(pd.to_numeric, errors='coerce')

# Разделяем X и Y
X = df.iloc[:, :-1].values  # Все столбцы, кроме последнего (признаки)
Y = df.iloc[:, -1].values.reshape(-1, 1)  # Последний столбец (целевая переменная)

# Нормализация данных (Z-score)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std  # Приведение данных к стандартному виду

# Добавляем столбец единиц для θ0
X = np.c_[np.ones(X.shape[0]), X]  # Добавляем столбец 1 в X

# Градиентный спуск
def compute_cost(X, Y, theta):
    m = len(Y)
    predictions = X @ theta
    errors = predictions - Y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient_descent(X, Y, theta, alpha, iterations):
    m = len(Y)
    cost_history = []

    for i in range(iterations):
        gradient = (1 / m) * (X.T @ (X @ theta - Y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, Y, theta))

    return theta, cost_history

# Параметры градиентного спуска
alpha = 0.1
iterations_list = [10, 100, 1000]

# Запускаем градиентный спуск для разных итераций
for iterations in iterations_list:
    theta = np.zeros((X.shape[1], 1))  # Инициализация нулями
    theta, cost_history = gradient_descent(X, Y, theta, alpha, iterations)

    print(f"\n# Итераций: {iterations}")
    print(f"Функция стоимости (округлённая): {round(cost_history[-1])}")
    print(f"Максимальное значение θ (округлённое): {round(np.max(theta))}")
