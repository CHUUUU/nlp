from multiprocessing import Process, Pipe

def do_something(duplex):
    duplex.send(['you', 'are', 'the', 'only', 'one'])
    duplex.close()

if __name__ == '__main__':
    '''
        Pipe() 함수는 파이프로 연결된 한 쌍의 연결 객체를 돌려주는데 기본적으로 양방향(duplex)입니다.
        Pipe() 가 반환하는 두 개의 연결 객체는 파이프의 두 끝을 나타냅니다. 
        각 연결 객체에는 (다른 것도 있지만) send() 및 recv() 메서드가 있습니다. 
        두 프로세스 (또는 스레드)가 파이프의 같은 끝에서 동시에 읽거나 쓰려고 하면 파이프의 데이터가 손상될 수 있습니다. 
        물론 파이프의 다른 끝을 동시에 사용하는 프로세스로 인해 손상될 위험은 없습니다.
        
        recv() = receive
    '''
    #
    connection1, connection2 = Pipe()
    p = Process(target=do_something, args=(connection1,))
    p.start()
    print(connection2.recv())   # ['you', 'are', 'the', 'only', 'one']
    p.join()