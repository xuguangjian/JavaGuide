

def cfunc( *args):
  print("有个{}参数".format(len(args)))
  print("第二个参数为：{}".format(args[1]))
cfunc("小甲鱼","java")