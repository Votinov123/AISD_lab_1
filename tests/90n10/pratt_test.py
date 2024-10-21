import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

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

np.random.seed(42)

sizes = np.arange(5000, 110001, 5000)

times_partial_sorted = []

for size in sizes:
    arr_partial_sorted = np.arange(size)  
    num_to_shuffle = size // 10  
    np.random.shuffle(arr_partial_sorted[-num_to_shuffle:])  
    start_time = time.time()
    pratt_sort(arr_partial_sorted)  
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