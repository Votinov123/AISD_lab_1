import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

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

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

np.random.seed(42)

sizes = np.arange(5000, 110001, 5000)

times_sorted = []

for size in sizes:
    arr_sorted = np.arange(size) 
    start_time = time.time()
    heap_sort(arr_sorted)
    end_time = time.time()
    times_sorted.append(end_time - start_time)
    print(size, end_time - start_time)

poly = PolynomialFeatures(degree=2)
N_b_poly_sorted = poly.fit_transform(sizes.reshape(-1, 1))
model_sorted = LinearRegression()
model_sorted.fit(N_b_poly_sorted, times_sorted)
y_predict_sorted = model_sorted.predict(N_b_poly_sorted)

plt.figure(figsize=(12, 6))
plt.scatter(sizes, times_sorted, label='Отсортированные массивы', color='blue', alpha=0.6, s=30)
plt.plot(sizes, y_predict_sorted, label='Регрессия', color='blue', linewidth=2)
plt.title('Время выполнения для отсортированных массивов')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (сек)')
plt.xlim(5000, 110000)
plt.grid()
plt.legend()
plt.show()