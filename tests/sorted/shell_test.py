import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

def shell_sort(arr):
    n = len(arr)
    gap = 1
    while gap < n // 2:
        gap *= 2

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

def generate_sorted_array(size):
    return np.arange(size)

np.random.seed(42)

sizes = np.arange(5000, 110001, 5000)

times_sorted = []

for size in sizes:
    arr_sorted = generate_sorted_array(size)
    start_time = time.time() 
    shell_sort(arr_sorted) 
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