import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

def selection_sort(arr):
    """Сортировка выбором."""
    lengh = len(arr)
    for i in range(lengh - 1):
        mas_min_ind = i
        for j in range(i + 1, lengh):
            if arr[j] < arr[mas_min_ind]:
                mas_min_ind = j
        arr[i], arr[mas_min_ind] = arr[mas_min_ind], arr[i]
    return arr

np.random.seed(42)

sizes = np.arange(200, 10001, 200)

times_descending = []

for size in sizes:
    arr_descending = np.arange(size, 0, -1)  
    start_time = time.time()  
    selection_sort(arr_descending)  
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
plt.xlim(200,10000) 
plt.grid() 
plt.legend() 
plt.show()