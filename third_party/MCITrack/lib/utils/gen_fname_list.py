import os
import argparse

parser = argparse.ArgumentParser(description='generate namelist')


args = parser.parse_args()


base_dir = "/home/cx/cx3/GOT10K/"  
result_file = "/home/cx/cx3/got10k.namelist"

assert (base_dir.endswith('/'))
fnames = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        path = os.path.join(root, file)
        if not os.path.exists(path):
            print("%s doesn't exist.")
        rela_path = path.replace(base_dir, "")
        fnames.append(rela_path)


with open(result_file, 'w') as fout:
    for name in fnames:
        fout.write(name + '\n')
