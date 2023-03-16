from re import X


def square(x):
    return x*x
square(3)

squareY=lambda y:y*y
squareY(4)
def boring(x):
    return ord(x)+10
list(map(boring(),"fishc"))