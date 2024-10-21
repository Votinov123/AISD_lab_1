def merge_sort(arr):
    if len(arr) > 1:
        i=0
        j=0
        k=0
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        merge_sort(left_half)  
        merge_sort(right_half)  
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
mas = [3, 4, 6, 17, 88, 1, 7, 11, 12, 8, 17, 67]
sorted_mas = merge_sort(mas)
print(sorted_mas)