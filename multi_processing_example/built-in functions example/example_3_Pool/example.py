from multiprocessing import Pool
import time
import os

start = time.time()

def do_something(a):
    print("sleeping 1 second")
    time.sleep(1)
    process = os.getpid()
    print("Process id : ", process)
    print('done sleeping')


if __name__ == "__main__":
    ps = [1, 2, 3, 4]
    p = Pool(processes=2)  # 사용 할 프로세스 수
    p.map(do_something, ps)
    p.close()
    p.join()

    print(str(round(time.time()-start, 2)) + "sec")