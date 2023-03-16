for i in range(3):
    print(i)
import copy

from torch import matrix_exp
x=[[1,2,3],[1,5,3],[2,3,3]]
y=copy.copy(x)
z=copy.deepcopy(x)
x[1][1]=0
y
print(x)
print(y)
print(z)
x=[i+4 for i in range(6)]
print(x)
A=[0]*3
for i in range(3):
    A[i]=[0]*3
A