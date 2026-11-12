import lmdb
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='generate lmdb')


args = parser.parse_args()


base_dir = "/home/cx/cx3/GOT10K/"  
result_file = "/home/cx/cx3/got10k.namelist"

namelist = [x.strip() for x in open(result_file).readlines()]
print('number:', len(namelist))

if base_dir.endswith("/"):
    lmdb_fname = base_dir[:-1] + '_lmdb'
else:
    lmdb_fname = base_dir + '_lmdb'
env = lmdb.open(lmdb_fname, map_size=1024 ** 4)
txn = env.begin(write=True)

for i, t in enumerate(tqdm(namelist)):
    if i % 100000 == 0:
        txn.commit()
        txn = env.begin(write=True)
    with open(os.path.join(base_dir, t), 'rb') as fin:
        txn.put(key=t.encode(), value=fin.read())

txn.commit()
env.close()
