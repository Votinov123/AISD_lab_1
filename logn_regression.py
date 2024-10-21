import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

# Функция сортировки Шелла
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr

# Функция быстрой сортировки
def quick_sort(arr):
    if len(arr) <= 1:
        return arr 
    pivot = arr[len(arr) // 2]
    left = []
    middle = []
    right = []
    for x in arr:
        if x < pivot:
            left.append(x)
        elif x == pivot:
            middle.append(x)
        else:
            right.append(x)
    return quick_sort(left) + middle + quick_sort(right)

# Генерация промежутков по Пратту
def pratt_sequence(n):
    gaps = []
    i = 0
    while True:
        for j in range(i + 1):
            gap = (2 ** i) * (3 ** j)
            if gap > n:
                return gaps
            gaps.append(gap)
        i += 1

# Функция сортировки с использованием последовательности Пратта
def pratt_sort(arr):
    n = len(arr)
    gaps = pratt_sequence(n)
    gaps.reverse()
    
    for gap in gaps:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
            
    return arr

# Функция сортировки слиянием
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Находим середину массива
        left_half = arr[:mid]  # Левый подмассив
        right_half = arr[mid:]  # Правый подмассив

        merge_sort(left_half)  # Рекурсивная сортировка левого подмассива
        merge_sort(right_half)  # Рекурсивная сортировка правого подмассива

        i = j = k = 0  # Индексы для левого, правого и основного массива

        # Слияние подмассивов
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Проверка, остались ли элементы в левом подмассиве
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        # Проверка, остались ли элементы в правом подмассиве
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
            
    return arr  

# Функция сортировки Шелла Хиббарда
def shell_sort_hibbard(arr):
    n = len(arr)
    gap = 1

    # Генерация промежутков Хиббарда
    while gap <= n // 2:
        gap = 2 * gap + 1

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

    return arr

# Функция для преобразования подмассива в кучу.
def heapify(arr, n, i):
    largest = i  
    left = 2 * i + 1  
    right = 2 * i + 2  

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Основная функция для сортировки массива с использованием пирамидальной сортировки.
def heap_sort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

# Установка фиксированного сида для генерации случайных чисел.
np.random.seed(42)

# Генерация размеров массивов от 5000 до 110000 с шагом 5000.
sizes = np.arange(5000, 110001, 5000)

# Измерение времени выполнения алгоритма для каждого размера массива.
times_shell, times_quick, times_pratt, times_merge, times_shell_hibbard, times_heap = [], [], [], [], [], []

for size in sizes:
    
   # Сортировка Шелла.
   arr_shell = np.random.randint(0, 109999, size)
   start_time = time.time()
   shell_sort(arr_shell)
   end_time = time.time()
   times_shell.append(end_time - start_time)

   # Быстрая сортировка.
   arr_quick = np.random.randint(0, 109999, size)
   start_time = time.time()
   quick_sort(arr_quick)
   end_time = time.time()
   times_quick.append(end_time - start_time)

   # Сортировка по Пратту.
   arr_pratt = np.random.randint(0, 109999, size)
   start_time = time.time()
   pratt_sort(arr_pratt)
   end_time = time.time()
   times_pratt.append(end_time - start_time)

   # Сортировка слиянием.
   arr_merge = np.random.randint(0, 109999, size)
   start_time = time.time()
   merge_sort(arr_merge)
   end_time = time.time()
   times_merge.append(end_time - start_time)

   # Сортировка Шелла Хиббарда.
   arr_shell_hibbard = np.random.randint(0, 109999, size)
   start_time = time.time()
   shell_sort_hibbard(arr_shell_hibbard)
   end_time = time.time()
   times_shell_hibbard.append(end_time - start_time)

   # Пирамидальная сортировка.
   arr_heap = np.random.randint(0, 109999, size)
   start_time = time.time()
   heap_sort(arr_heap)
   end_time = time.time()
   times_heap.append(end_time - start_time)

# Преобразуем в массив NumPy для удобства.
y_shell     = np.array(times_shell)
y_quick     = np.array(times_quick)
y_pratt     = np.array(times_pratt)
y_merge     = np.array(times_merge)
y_shell_hibbard     = np.array(times_shell_hibbard)
y_heap      = np.array(times_heap)

# Полиномиальная регрессия (2-й степени) для всех алгоритмов сортировки и предсказание значений y для натуральных чисел.
def plot_regression(sizes, y_times, label, color):
    poly_features      = PolynomialFeatures(degree=2)
    N_poly             = poly_features.fit_transform(sizes.reshape(-1, 1))
    
    model_regression   = LinearRegression()
    model_regression.fit(N_poly , y_times)

    y_predict_regression= model_regression.predict(N_poly)

    plt.plot(sizes , y_predict_regression , label=label , color=color)

# Создание графика с регрессиями для всех алгоритмов сортировки.
plt.figure(figsize=(12 ,6))
plot_regression(sizes , y_shell         , 'Shell Sort (Регрессия)' , 'cyan')
plot_regression(sizes , y_quick         , 'Quick Sort (Регрессия)' , 'purple')
plot_regression(sizes , y_pratt         , 'Pratt Sort (Регрессия)' , 'orange')
plot_regression(sizes , y_merge         , 'Merge Sort (Регрессия)' , 'blue')
plot_regression(sizes , y_shell_hibbard , 'Shell Hibbard (Регрессия)' , 'green')
plot_regression(sizes , y_heap          , 'Heap Sort (Регрессия)' , 'red')

# Настройка графика.
plt.title('Время выполнения различных алгоритмов сортировки (Регрессия)')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (сек)')
plt.xlim(5000 ,110000)  
plt.ylim(0 , max(max(y_shell), max(y_quick), max(y_pratt), max(y_merge), max(y_shell_hibbard), max(y_heap)) * 1.1)  
plt.axhline(0 , color='black', linewidth=0.5 , ls='--')
plt.axvline(0 , color='black', linewidth=0.5 , ls='--')
plt.grid()
plt.legend()
plt.show()