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

my_list = [3, 4, 6, 17, 88, 1, 7, 11, 12, 8, 17, 67]
sorted_list = shell_sort(my_list)
print(sorted_list)