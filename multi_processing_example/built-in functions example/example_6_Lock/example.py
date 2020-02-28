from multiprocessing import Process, Lock

def do_something(lock, num):
    lock.acquire()
    print("hello ", str(num))
    lock.release()

if __name__ == '__main__':
    '''
        예를 들어 한 번에 하나의 프로세스만 표준 출력으로 인쇄하도록 록을 사용할 수 있습니다
        록을 사용하지 않으면 다른 프로세스의 출력들이 모두 섞일 수 있습니다.
        
        확실히 잘 모르겠음
    '''
    lock = Lock()
    for num in range(10):
        Process(target=do_something, args=(lock, num)).start()

    # hello    0
    # hello    2
    # hello    3
    # hello    1
    # hello    5
    # hello    7
    # hello    9
    # hello    8
    # hello    6
    # hello    4
