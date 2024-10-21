import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

# Функция сортировки пузырьком
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Функция сортировки выбором
def selection_sort(arr):
    length = len(arr)
    for i in range(length - 1):
        min_index = i
        for j in range(i + 1, length):
            if arr[j] < arr[min_index]:
                min_index = j
        # Обмен значениями
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

# Функция сортировки вставками
def insertion_sort(arr):
    length = len(arr)
    for i in range(1, length):  # Начинаем с 1, так как элемент на 0 уже "отсортирован"
        key = arr[i]
        j = i - 1
        # Перемещаем элементы, которые больше key, на одну позицию вперед
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key  # Вставляем ключевой элемент на правильную позицию
    return arr

# Установка фиксированного сида для генерации случайных чисел
np.random.seed(42)

# Генерация размеров массивов от 200 до 10000 с шагом 200
sizes = np.arange(200, 10001, 200)  # Размеры массивов с шагом 200

# Измерение времени выполнения алгоритма для каждого размера массива
times_bubble = []
times_selection = []
times_insertion = []

for size in sizes:
    arr_bubble = np.random.randint(0, 10000, size)  # Генерация случайного массива для пузырька
    start_time = time.time()
    bubble_sort(arr_bubble)
    end_time = time.time()
    times_bubble.append(end_time - start_time)

    arr_selection = np.random.randint(0, 10000, size)  # Генерация случайного массива для выбора
    start_time = time.time()
    selection_sort(arr_selection)
    end_time = time.time()
    times_selection.append(end_time - start_time)

    arr_insertion = np.random.randint(0, 10000, size)  # Генерация случайного массива для вставки
    start_time = time.time()
    insertion_sort(arr_insertion)
    end_time = time.time()
    times_insertion.append(end_time - start_time)

# Преобразуем в массив NumPy для удобства
y_bubble = np.array(times_bubble)
y_selection = np.array(times_selection)
y_insertion = np.array(times_insertion)

# Полиномиальная регрессия (2-й степени) для пузырьковой сортировки
poly_bubble = PolynomialFeatures(degree=2)
N_b_poly = poly_bubble.fit_transform(sizes.reshape(-1, 1))
model_bubble = LinearRegression()
model_bubble.fit(N_b_poly, y_bubble)
y_bubble_predict = model_bubble.predict(N_b_poly)

# Полиномиальная регрессия (2-й степени) для сортировки выбором
poly_selection = PolynomialFeatures(degree=2)
N_s_poly = poly_selection.fit_transform(sizes.reshape(-1, 1))
model_selection = LinearRegression()
model_selection.fit(N_s_poly, y_selection)
y_selection_predict = model_selection.predict(N_s_poly)

# Полиномиальная регрессия (2-й степени) для сортировки вставками
poly_insertion = PolynomialFeatures(degree=2)
N_i_poly = poly_insertion.fit_transform(sizes.reshape(-1, 1))
model_insertion = LinearRegression()
model_insertion.fit(N_i_poly, y_insertion)
y_insertion_predict = model_insertion.predict(N_i_poly)

# Создание графика
plt.figure(figsize=(12, 6))
plt.plot(sizes, y_bubble_predict, label='Bubble Sort (Регрессия)', color='red', linewidth=2)
plt.plot(sizes, y_selection_predict, label='Selection Sort (Регрессия)', color='green', linewidth=2)
plt.plot(sizes, y_insertion_predict, label='Insertion Sort (Регрессия)', color='blue', linewidth=2)

# Настройка графика
plt.title('Время выполнения сортировочных алгоритмов (Регрессия)')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (сек)')
plt.xlim(200, 10000)  
plt.ylim(0, max(max(y_bubble), max(y_selection), max(y_insertion)) * 1.1)  
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()