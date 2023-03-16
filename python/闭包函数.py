def funA():
    x=800
    def funB():
        print(x)
    return funB
funA()
funA()()
funny=funA()
funny()