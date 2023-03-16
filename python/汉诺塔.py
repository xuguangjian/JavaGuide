def han(n,x,y,z):
    if n==1:
        print(x,"-->",z)
    else:
        han(n-1,x,z,y)
        print(x,"-->",y)
        han(n-1,y,x,z)
n=int(input("层数："))
han(n,'A','B','C')