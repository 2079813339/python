def pow(shu,cifang):
   x = shu**cifang
   return x
def gcd(shu1,shu2):
   if shu1 > shu2:
       while shu2!= 0:
           yushu = shu1%shu2
           shu1 = shu2
           shu2 = yushu
           return shu2
   else:
       while shu1 != 0:
           yushu = shu2%shu1
           shu2 = shu1
           shu1 = yushu
           return shu1
def lcm(shu1,shu2):
   yushu = gcd(shu1,shu2)
   gobeishu = int((shu1*shu2)/yushu)
   return gobeishu
def fabs(shu):
   if shu == 0:
       return 0
   elif shu>0:
       return shu
   else:
       return (-shu)
def floor(shu):
   if shu>=0:
       return int(shu)
   else :
       return int(shu)-1
def ceil(shu):
       return int(shu)+1
def factorial(shu):
   val = 1
   if shu ==0:
       return shu
   else:
       y = 1
       while y<=shu:
           val = val*y
           y +=1
       return val
def fsum():
    x = [int(i) for i in input("请输入数字(空格隔开):").split(" ")]
    s = sum(x)
    return s
mingling = input("请输入命令(pow,gcd,lcm,floor,factorial,fsum):")
while mingling != "esc":
    print(mingling)
    if mingling == "pow":
        shu1 = int(input("请输入数字："))
        shu2 = int(input("请输入次方:"))
        print(pow(shu1,shu2))
    elif mingling =="gcd":
        shu1 = int(input("请输入第一个数字:"))
        shu2 = int(input("请输入第二个数字:"))
        print(gcd(shu1,shu2))
    elif mingling =="lcm":
        shu1 = int(input("请输入第一个数字:"))
        shu2 = int(input("请输入第二个数字:"))
        print(lcm(shu1,shu2))
    elif mingling == "fabs":
        shu1 = int(input("请输入一个数字:"))
        print(fabs(shu1))
    elif mingling == "floor":
        shu1 = int(input("请输入一个数字:"))
        print(floor(shu1))
    elif mingling == "ceil":
        shu1 = int(input("请输入一个数字:"))
        print(ceil(shu1))
    elif mingling =="factorial":
        shu1 = int(input("请输入一个数字:"))
        print(factorial(shu1))
    elif mingling == "fsum":
        print(fsum())
    else:
        break
    mingling = input("请输入命令(pow,gcd,lcm,floor,factorial,fsum(esc为退出):")
