from cgi import print_arguments


def temp(c):
    f=c*1.8+32
    return f
c=float(input("请输入摄氏度："))
f=temp(c)
print("转换成华氏温度："+str(f))