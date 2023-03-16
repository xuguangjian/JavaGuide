def fun():
    for i in range(10):
        print(i)
        
def cfunc(name):
    for i in range(3):
        print(f"love {name}")
cfunc("python")
fun()

def cfunc(name,times):
    for i in range(times):
        print(f"love {name}")
cfunc("java",3)

def dix(x,y):
    if y==0:
        return "除数不能为零！"
    else:
        return x/y
dix(4,2)