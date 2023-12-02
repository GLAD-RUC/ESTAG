import amc_parser as amc
from glob import glob
import numpy as np
import pickle as pkl


BASE_DIR = 'motion/subject_102'
# asf_name = '35.asf'
# BASE_DIR = '.'
asf_name = '102.asf'

joints = amc.parse_asf(BASE_DIR + '/' + asf_name)
joints['root'].get_name_to_idx()
edges = joints['root'].output_edges()
print('All edges:', len(edges))
print(edges)

all_X = []

for amc_name in glob(BASE_DIR + '/*.amc.txt'):
    print(amc_name)
    motions = amc.parse_amc(amc_name)
    if amc_name.split('.')[-2].split('_')[-1] == '10':
        print(amc_name, ' is the special case!!!')
        motions = motions[6:]
    T = len(motions)
    print('Frame:', T)
    XX = []
    for i in range(T):
        joints['root'].set_motion(motions[i])
        X = joints['root'].output_coord()
        XX.append(X)
    XX = np.array(XX)
    print(XX.shape)
    all_X.append(XX)

with open('motion/motion_basketball.pkl', 'wb') as f:
    pkl.dump((edges, all_X), f)

print('Saved to motion_basketball.pkl')



