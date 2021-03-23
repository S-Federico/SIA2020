#Determinare l’output del programma seguente:
def fun(a):
    return a[2:]

a=[1,2,3,4,5]
b=a
b[3]=6
c=fun(a)
c[2]=3
print (c)

#3,6,3