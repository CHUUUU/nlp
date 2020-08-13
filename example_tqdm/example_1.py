from tqdm import tqdm
import time

# for i in tqdm(range(100), desc="tqdm example", mininterval=0.01): 
#     time.sleep(0.1)
# tqdm example:  79%|████████████████████████████████████████████████████████████████████████████████████████▍                       | 79/100 [00:08<00:02,  9.84it/s]


# t = tqdm(range(100))
# for i in t:
#     t.set_description("index : ", i)
# index : : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 12549.53it/s]


t = tqdm(range(100))
for i in t:
    for j in range(100):
        t.set_description("i : {}  j : {}  i + j : {}  i * j : {}".format(i, j, i+j, i*j))
        time.sleep(0.001)
# i : 83  j : 84  i + j : 167  i * j : 6972:  83%|████████████████████████████████████████████████████████████████████▉              | 83/100 [00:15<00:03,  5.24it/s] 