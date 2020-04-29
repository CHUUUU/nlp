## iterables
# 모든 값을 메모리에 담아둔 상태로 한개씩 꺼내서 순환

# example 1
mylist = [1, 2, 3]
for i in mylist:
    print(i)

# example 2
mylist = [x * x for x in range(3)]
for i in mylist:
   print(i)

#################################################################################################

## Generators
# 모든 값을 메모리에 담아두지 않고 그때그때 값을 생성(Generator)하여 순환

# example 1
mygenerator = (x * x for x in range(3))
for i in mygenerator:
    print(i)

# example 2
# yield 를 더이상 안만날 때까지 for 문에서 return
# 보통 함수는 return 을 만나면 끝내지만
# generator 는 for 문에서 yield 으로 return 해주다가 yield 가 더이상 안나오면 함수가 종료
# 즉 for 문을 계속 돌면서 return 을 시킬 수 있으면서, 메모리 관리까지 해주는 아주 좋은 문법임
# 함수로 따로 만들어서, 안에 커스텀 시켜서 내 맘대로 반환받기 쉽게 할 수 있어 좋음
def createGenerator():
    mylist = range(3)
    for i in mylist:
        yield i * i
mygenerator = createGenerator() # 제너레이터 생성
for i in mygenerator:
    print(i)


# 1
# 2
# 3
# 0
# 1
# 4
# 0
# 1
# 4
# 0
# 1
# 4