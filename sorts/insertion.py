mas=[3,4,6,17,88,1,7,11,12, 8,17,67]
buf=0
lengh=len(mas)
for i in range (lengh -1 ):
    if (mas[i] > mas[i+1]):
        for j in range (i+1,0,-1):
            if (mas[j] < mas[j-1]):
                buf=mas[j]
                mas[j]=mas[j-1]
                mas[j-1]=buf
            else: break
print(mas)
