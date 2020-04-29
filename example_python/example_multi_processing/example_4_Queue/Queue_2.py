from multiprocessing import Process, Queue

def f(q):
    for i in range(0,100):
        print("come on baby")
    q.put([42, None, 'hello'])


if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    for j in range(0, 2000):
        if j == 100:
            print(q.get())
        print(j)


# 특징 -=> main process 가 subprocess 나올때까지 기다려줌

# 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13

# ...
# ...

# 88
# 89
# 90
# 91
# 92
# 93
# 94
# 95
# 96
# 97
# 98
# 99
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# come on baby
# [42, None, 'hello']
# 100
# 101
# 102
# 103
# 104
# 105
# 106
# 107
#
# ...
# ...
#
# 1988
# 1989
# 1990
# 1991
# 1992
# 1993
# 1994
# 1995
# 1996
# 1997
# 1998
# 1999
