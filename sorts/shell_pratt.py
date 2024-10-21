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
    print(gaps)
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

arr = [3, 4, 6, 17, 88, 1, 7, 11, 12, 8, 17, 67]
sorted_arr_pratt = pratt_sort(arr.copy())
print(sorted_arr_pratt)