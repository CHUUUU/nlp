from multiprocessing import Manager, Process
import parmap
import time
import os

start = time.time()

def do_something(share_dic, input_arg):
    process = os.getpid()
    print("Process id : ", process)
    share_dic[1] = '1'
    share_dic['2'] = 2
    share_dic[0.25] = None
    input_arg.reverse()

if __name__ == "__main__":
    manager = Manager()  # manager 는 Process 간의 데이터 공유가 가능하게 한다.
    share_dic = manager.dict()
    input_arg = manager.list(range(10))

    p = Process(target=do_something,
                args=(share_dic, input_arg))

    p.start()
    p.join()

    print(str(round(time.time()-start, 2)) + "sec")
    print(share_dic)
    print(input_arg)
    
#     Process id : 14664
#     Process id : 2472
#     0.27sec
#     {1: '1', '2': 2, 0.25: None, 100: 100}
#     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 100]
