from operator import mod
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

parser.add_argument('--rs', type=int, default=100, metavar='N',
                    help='rollout steps')



args = parser.parse_args()

####################
# args.vel = False

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
    model_list = ['estag','gnn', 'stgcn', 'egnn']    
    # model_list = ['estag']    

    res_99 = {}
    res_199 = {}
    
    
    for mol in mol_list[0:8]:
        plt.figure()
        # plt.rcParams['font.family'] = 'Times New Roman'
        
        res_99.setdefault(mol, [])
        res_199.setdefault(mol, [])

        dataset = MD17Dataset(partition='test', max_samples=2000, data_dir=args.data_dir,
                                    molecule_type=mol, delta_frame=args.delta_frame, num_past=args.num_past)
        
        x = torch.FloatTensor(dataset.x)
        n_samples, n_nodes=x.shape[:2]

        pivot = np.random.randint(args.num_past*args.delta_frame, n_samples-args.rs*args.delta_frame, 1)[0]
        print('pivot: ', pivot)

        past_idx = [pivot-i*args.delta_frame for i in range(args.num_past, 0, -1)]
        # print(past_idx)
        # assert False

        loc = x[past_idx].to(device)

        edge_attr, charges = dataset.edge_attr, dataset.mole_idx.unsqueeze(-1)
        edge_attr, charges = edge_attr.to(device), charges.to(device)
        
        edges = dataset.get_edges(1, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]
        

        for model_name in tqdm(model_list):
            # if model_name not in ['estag']:
            #     continue

            # exp_path = f"temp/exp_{mol}/{model_name}"
            exp_path = f"logs/md17_logs/exp_{mol}/{model_name}"
            model=torch.load(f"{exp_path}/saved_model.pth", map_location=device)
            model.eval()

            loc_temp = loc.clone()

            l = []
            with torch.no_grad():
                for r in range(args.rs):
                    print('*********round: ', r)
                    if model_name == 'egnn':
                        loc_pred = model(charges, loc_temp, edges, edge_attr)
                    elif model_name == 'estag':
                        loc_pred = model(charges, loc_temp, edges, edge_attr)
                    elif model_name == 'stgcn':
                        feature = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc_temp), dim=-1)
                        node = feature.permute(1,0,2).reshape(1,n_nodes,feature.shape[0],feature.shape[2])
                        Adj = dataset.A.to(device)
                        loc_pred = loc_temp[-1] + model(Adj, node).reshape(-1,3)
                    elif model_name == 'gnn':
                        nodes = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc_temp), dim=-1)
                        
                        loc_pred = model(nodes, edges, edge_attr)

                    next_idx = pivot + r * args.delta_frame
                    loc_end = x[next_idx].to(device)

                    # print(loc_pred)
                    # print(loc_end)
                    # sleep(5)

                    loss = loss_mse(loc_pred, loc_end)
                    l.append(loss.item())

                    if r == 99:
                        res_99[mol].append(loss.item())
                    if r == 199:
                        res_199[mol].append(loss.item())

                    loc_temp = torch.cat((loc_temp[1:], loc_pred.unsqueeze(0)), dim=0)
                    # print(loc_pred)
                    # sleep(0.5)
                    print(loss.item())

                    # break

            loss = torch.FloatTensor(l)

            if model_name == 'egnn':
                label = 'ST_EGNN'
            else:
                label = model_name.upper()
            plt.plot(loss*1000,label=label, linestyle='--')


        plt.xlabel('Step', fontsize=12)
        plt.ylabel(r'MSE ($\times 10^{-3}$)', fontsize=12)
        plt.xticks(np.arange(start=0, stop=len(loss), step=2))
        plt.yscale('log')
        plt.minorticks_off()
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.title(mol.upper(), fontsize=12)
        plt.legend(loc=4, fontsize=15)

        plt.savefig(f'./figures/rollout_{mol}.pdf')
        # break

    print('***************loss 99:')
    all_loss = np.array(list(res_99.values())) * 100
    print(all_loss.T)

    print('***************loss 199:')
    all_loss = np.array(list(res_199.values())) * 100
    print(all_loss.T)





if __name__ == "__main__":
    main()




