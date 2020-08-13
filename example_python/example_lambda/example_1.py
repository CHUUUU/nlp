print(lambda x: 0)
# <function <lambda> at 0x000001FEE11179D8>

print(map(lambda x: 0, [1, 2, 3]))
# <map object at 0x000001FEE11452C8>

print(list(map(lambda x: 0, [1, 2, 3])))
# [0, 0, 0]

print(list(map(lambda x, y: x*y, [1, 2, 3], [1, 2, 3])))  
# map 은 반복, 리스트를 인자로 받음
# r = map(function, iterable, ...)
# [1, 4, 9]

f = lambda x, y: x + y
print(f(4, 4))
# 8


# 파이썬에서 "lambda" 는 런타임에 생성해서 사용할 수 있는 익명 함수 입니다. 
# 이것은 함수형 프로그래밍 언어에서 lambda와 정확히 똑같은 것은 아니지만,  
# 파이썬에 잘 통합되어 있으며 filter(), map(), reduce()와  같은 
# 전형적인 기능 개념과 함께 사용되는 매우 강력한 개념입니다.
# lambda는 쓰고 버리는 일시적인 함수 입니다. 
# 즉, 간단한 기능을 일반적인 함수와 같이 정의해두고 쓰는 것이 아니고 필요한 곳에서 즉시 사용하고 버릴 수 있습니다.
# 람다 정의에는 "return"문이 포함되어 있지 않습니다. 

# https://offbyone.tistory.com/73