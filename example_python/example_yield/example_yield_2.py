
# yield from 반복가능한 객체
# yield from 이터레이터
# yield from 제너레이터

# example 1
def number_generator():
    x = [1, 2, 3]
    yield from x

for i in number_generator():
    print(i)

# 1
# 2
# 3


# example 2
g = number_generator()
print(next(g))
print(next(g))
print(next(g))

# 1
# 2
# 3