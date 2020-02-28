from multiprocessing import Process, Queue

def do_something(q):
    q.put(['you', 'are', 'the', 'only', 'one'])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=do_something, args=(q,))
    p.start()
    print(q.get())  # ['you', 'are', 'the', 'only', 'one']
    p.join()