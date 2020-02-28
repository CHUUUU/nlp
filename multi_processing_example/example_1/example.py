import multiprocessing
import time

start = time.perf_counter()

def do_something():
    print("sleeping 1 second")
    time.sleep(1)
    print('done sleeping')

do_something()
do_something()

end = time.perf_counter()
print(str(round(end-start, 2)) + "sec")