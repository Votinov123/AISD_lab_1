def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j] 
    return arr
arr = [3, 4, 6, 17, 88, 1, 7, 11, 12, 8, 17, 67]
sorted_arr = bubble_sort(arr)
print(sorted_arr)