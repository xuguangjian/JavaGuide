for n in range(2,10):
    for x in range(2,n):
        if n%x==0:
            print(n, "=",x,"*",n//x)
            break
    else:
        print(n,"素数")


        
n = int(input("please enter the number:"))
for i in range(2, n):
    if n % i == 0:
        print(" %d is not a prime number!" % n)
        break
else:
    print(" %d is a prime number!" % n)






for i in range(2, 2):
    print(i)
for i in range(2):
    print(i)
n = int(input("请输入一个数字: "))
for i in range(2, n):
    print(i)