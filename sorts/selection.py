mas=[3,4,6,17,88,1,7,11,12, 8,17,67]
buf=0
lengh=len(mas)
for i in range(lengh-1):
    mas_min_ind=i
    for j in range(i+1, lengh):
        if mas[j]<mas[mas_min_ind]:
            mas_min_ind=j
    buf=mas[i]
    mas[i]=mas[mas_min_ind]
    mas[mas_min_ind]=buf     
print(mas)    
        


