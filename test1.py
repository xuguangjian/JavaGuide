"""用python设计一个游戏"""
import random
count=3
answer=random.randint(1,10)
while count>0:
    temp=input("不妨猜一下数字：")
    guess=float(temp)
    if guess==answer:
        print("对了!")
        print("hello PYTHON")
        break
    else:
        if guess<answer:
            print("小了")
        else:
            print("大了")
        count=count-1

print("游戏结束")