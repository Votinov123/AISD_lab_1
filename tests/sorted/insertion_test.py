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

np.random.seed(42)

sizes = np.arange(200, 10001, 200)

times_sorted = []

for size in sizes:
    arr_sorted = np.arange(size) 
    start_time = time.time() 
    insertion_sort(arr_sorted)
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
plt.xlim(200, 10000)
plt.grid()
plt.legend()
plt.show()