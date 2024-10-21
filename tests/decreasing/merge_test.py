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

times_descending = []

for size in sizes:
    arr_descending = np.arange(size, 0, -1)  
    start_time = time.time()  
    merge_sort(arr_descending)  
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