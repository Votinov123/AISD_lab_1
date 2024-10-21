import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
            
    return arr  

np.random.seed(42)

sizes = np.arange(5000, 110001, 5000)
times = []

for size in sizes:
    arr = np.random.randint(0, 10000, size)
    start_time = time.time()
    merge_sort(arr)
    end_time = time.time()
    times.append(end_time - start_time)
    print(size, end_time - start_time)

y = np.array(times)

poly = PolynomialFeatures(degree=2)
N_b_poly = poly.fit_transform(sizes.reshape(-1, 1))

model = LinearRegression()
model.fit(N_b_poly, y)

y_predict = model.predict(N_b_poly)

plt.figure(figsize=(12, 6))
plt.scatter(sizes, y, label='Экспериментальные данные', color='blue', alpha=0.6, s=30)
plt.plot(sizes, y_predict, label='Функция сложности (регрессия)', color='red', linewidth=2)
plt.title('Время выполнения алгоритма в зависимости от размера массива')
plt.xlabel('Размер массива')
plt.ylabel('Время выполнения (сек)')
plt.xlim(5000, 110000)
plt.ylim(0, max(y) + max(y) * 0.1)
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()