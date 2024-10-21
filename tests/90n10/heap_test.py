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

times_partial_sorted = []

for size in sizes:
    arr_partial_sorted = np.arange(size)  
    num_to_shuffle = size // 10  
    np.random.shuffle(arr_partial_sorted[-num_to_shuffle:])  
    start_time = time.time()
    heap_sort(arr_partial_sorted)
    end_time = time.time()
    times_partial_sorted.append(end_time - start_time)
    print(size, end_time - start_time)

poly = PolynomialFeatures(degree=2)
N_b_poly_partial_sorted = poly.fit_transform(sizes.reshape(-1, 1))
model_partial_sorted = LinearRegression()
model_partial_sorted.fit(N_b_poly_partial_sorted, times_partial_sorted)
y_predict_partial_sorted = model_partial_sorted.predict(N_b_poly_partial_sorted)

plt.figure(figsize=(12, 6))
plt.scatter(sizes, times_partial_sorted, label='90% отсортированные массивы', color='orange', alpha=0.6, s=30)
plt.plot(sizes, y_predict_partial_sorted, label='Регрессия', color='orange', linewidth=2)
plt.title('Время выполнения для массивов, отсортированных на 90%')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (сек)')
plt.xlim(5000, 110000)
plt.grid()
plt.legend()
plt.show()