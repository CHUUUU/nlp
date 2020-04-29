def number_generator(stop):
    n = 0
    while n < stop:
        yield n
        n += 1

def gen():
    yield from number_generator(3)

for i in gen():
    print(i)

# 0
# 1
# 2


g = gen()
print(next(g))
print(next(g))
print(next(g))

# 0
# 1
# 2