from multiprocessing import Process
import time
import os

start = time.time()

def do_something():
    print("sleeping 1 second")
    time.sleep(1)
    process = os.getpid()
    print("Process id : ", process)
    print('done sleeping')


if __name__ == "__main__":
    ps =[]
    for i in range(2):
        p = Process(target=do_something)
        ps.append(p)
        p.start()

    for p in ps:
        p.join()

    print(str(round(time.time()-start, 2)) + "sec")