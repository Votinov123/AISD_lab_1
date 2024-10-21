import random
def is_sorted(mas):
    n = len(mas)
    for i in range(n - 1):
        if mas[i] > mas[i + 1]:
            return False
    return True
def shuffle(mas):
    n = len(mas)
    for i in range(n):
        j = random.randint(0, n - 1)
        mas[i], mas[j] = mas[j], mas[i]
    return mas
mas = [1,2,5,3,4]
while not is_sorted(mas): shuffle(mas)
print(mas)