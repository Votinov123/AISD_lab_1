import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

def insertion_sort(arr):
    length = len(arr)
    for i in range(1, length):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def generate_partially_sorted_array(size):
    arr = np.arange(size)
    num_to_shuffle = size // 10
    np.random.shuffle(arr[-num_to_shuffle:])  
    return arr

np.random.seed(42)

sizes = np.arange(200, 10001, 200)

times_partial_sorted = []

for size in sizes:
    arr_partial_sorted = generate_partially_sorted_array(size)
    start_time = time.time()
    insertion_sort(arr_partial_sorted)
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
plt.xlim(200, 10000)
plt.grid()
plt.legend()
plt.show()