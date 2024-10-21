import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j] 
    return arr

np.random.seed(42)

sizes = np.arange(200, 10001, 200)

times = []

for size in sizes:
    arr = np.random.randint(0, 10000, size) 
    start_time = time.time() 
    bubble_sort(arr)  
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
plt.xlim(200, 10000) 
plt.ylim(0, max(y) + max(y)*0.1) 
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()