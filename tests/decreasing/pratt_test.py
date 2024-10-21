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

times_descending = []

for size in sizes:
    arr_descending = np.arange(size, 0, -1)  
    start_time = time.time()
    pratt_sort(arr_descending)  
    end_time = time.time()
    times_descending.append(end_time - start_time)
    print(size, end_time - start_time)

poly_descending = PolynomialFeatures(degree=2)
N_b_poly_descending = poly_descending.fit_transform(sizes.reshape(-1, 1))
model_descending = LinearRegression()
model_descending.fit(N_b_poly_descending, times_descending)
y_predict_descending = model_descending.predict(N_b_poly_descending)

plt.figure(figsize=(12,6)) 
plt.scatter(sizes, times_descending, label='Убывающие массивы', color='red', alpha=0.6, s=30) 
plt.plot(sizes, y_predict_descending, label='Регрессия', color='red', linewidth=2) 
plt.title('Время выполнения для убывающих массивов') 
plt.xlabel('Размер массива') 
plt.ylabel('Время выполнения (сек)') 
plt.xlim(5000, 110000) 
plt.grid() 
plt.legend() 
plt.show()