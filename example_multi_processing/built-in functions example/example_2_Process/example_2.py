from multiprocessing import Process
import time
import os

start = time.time()

def count(num):
    process = os.getpid()
    for i in range(num):
        print("Process id : ", process, "--", i)

if __name__ == "__main__":
    num_arr = [200, 200]
    ps = []

    # process 시작
    for num in num_arr:
        p = Process(target=count, args=(num,))
        ps.append(p)
        p.start()  # count 함수 시작

    # 프로세스 종료 대기
    for p in ps:
        p.join()

print(str(round(time.time() - start, 2)) + " sec")