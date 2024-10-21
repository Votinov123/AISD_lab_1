import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

def shell_sort_hibbard(arr):
    n = len(arr)
    gap = 1

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
        gap = (gap - 1) // 2  

    return arr

np.random.seed(42)

sizes = np.arange(5000, 110001, 5000)

times_descending = []

for size in sizes:
    arr_descending = np.arange(size, 0, -1)  
    start_time = time.time()
    shell_sort_hibbard(arr_descending)  
    end_time = time.time()
    times_descending.append(end_time - start_time)
    print(size, end_time - start_time)

poly = PolynomialFeatures(degree=2)
N_b_poly_descending = poly.fit_transform(sizes.reshape(-1, 1))
model_descending = LinearRegression()
model_descending.fit(N_b_poly_descending, times_descending)
y_predict_descending = model_descending.predict(N_b_poly_descending)

plt.figure(figsize=(12,6)) 
plt.scatter(sizes,times_descending,label='Убывающие массивы',color='red',alpha=0.6,s=30) 
plt.plot(sizes,y_predict_descending,label='Регрессия',color='red',linewidth=2) 
plt.title('Время выполнения для убывающих массивов') 
plt.xlabel('Размер массива') 
plt.ylabel('Время выполнения (сек)') 
plt.xlim(5000,110000) 
plt.grid() 
plt.legend() 
plt.show()