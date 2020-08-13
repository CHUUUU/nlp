from tqdm import tqdm
import time

for i in tqdm(range(100), desc="tqdm example", mininterval=0.01): #
    time.sleep(0.1)

# tqdm example:  79%|████████████████████████████████████████████████████████████████████████████████████████▍                       | 79/100 [00:08<00:02,  9.84it/s]


t = tqdm(range(100))
for i in t:
    t.set_description("index : ", i)
# index : : 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 12549.53it/s]