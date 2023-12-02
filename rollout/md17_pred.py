import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

import argparse
from math import exp
import torch
import torch.utils.data
from md17.dataset import MD17Dataset
from models.model import *
import os
from torch import nn, optim
import json

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='STAG')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--outf', type=str, default='logs/md17_logs', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--num_past', type=int, default=10,
                    help='Number of length of whole past time series.')


parser.add_argument('--max_training_samples', type=int, default=500, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--data_dir', type=str, default='md17',
                    help='Data directory.')

parser.add_argument('--exp_name', type=str, default='exp_10000', metavar='N', help='experiment_name')
parser.add_argument('--delta_frame', type=int, default=10,
                    help='Number of frames delta.')


args = parser.parse_args()

####################

args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)



def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    mol_list =  ['aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil']
    model_list = ['estag', 'egnn']    
    # model_list = ['estag']    

    
    for mol in mol_list:        
        dataset = MD17Dataset(partition='test', max_samples=2000, data_dir=args.data_dir,
                                    molecule_type=mol, delta_frame=args.delta_frame, num_past=args.num_past)
        
        x = torch.FloatTensor(dataset.x)
        n_samples, n_nodes=x.shape[:2]

        pivot = 128047
        # pivot = np.random.randint(args.num_past*args.delta_frame, n_samples-args.delta_frame, 1)[0]
        print('pivot: ', pivot)

        past_idx = [pivot-i*args.delta_frame for i in range(args.num_past, 0, -1)]

        loc = x[past_idx].to(device)

        
        edge_attr, charges = dataset.edge_attr, dataset.mole_idx.unsqueeze(-1)
        edge_attr, charges = edge_attr.to(device), charges.to(device)

        
        edges = dataset.get_edges(1, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]
        

        for model_name in model_list:
            print('********', mol, model_name)
            exp_path = f"logs/md17_logs/exp_{mol}/{model_name}"
            model=torch.load(f"{exp_path}/saved_model.pth", map_location=device)
            model.eval()

            with torch.no_grad():
                if model_name == 'egnn':
                    loc_pred = model(charges, loc, edges, edge_attr)
                elif model_name == 'estag':
                    loc_pred = model(charges, loc, edges, edge_attr)

                next_idx = pivot
                loc_end = x[next_idx].to(device)

                print(loc_pred)

                loss = loss_mse(loc_pred, loc_end)
                print(loss.item())



if __name__ == "__main__":
    main()




