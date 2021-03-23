#Determinare l’output del programma seguente:
a=['a','b',['b','c'],1,2,3]
del a[0]
a[1][0]='a'
c=a[2:4]
d=a[1]
e=c+d
print(e)

# 1,2,a,c