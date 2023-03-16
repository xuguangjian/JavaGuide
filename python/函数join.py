def cfunc(s,vt,c):
    return "".join((c,vt,s))
cfunc("我","打了","小甲鱼")
cfunc(c="我",vt="打了",s="小甲鱼")


def cfunc(s,vt,c="小甲鱼"):
    return "".join((c,vt,s))
cfunc("我","打了")