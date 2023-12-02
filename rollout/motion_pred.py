import sys
import os.path as osp


sys.path.append(osp.dirname(osp.dirname(__file__)))

import argparse
import torch
import torch.utils.data
from motion.dataset import MotionDataset
from models.model import *

import random
import numpy as np
from tqdm import tqdm



parser = argparse.ArgumentParser(description='STAG')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--data_dir', type=str, default='motion',
                    help='Data directory.')

parser.add_argument('--num_past', type=int, default=10,
                    help='Number of length of whole past time series.')
parser.add_argument('--delta_frame', type=int, default=5,
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

    cases =  ['walk', 'basketball']
    model_list = ['stag_egnn', 'egnn']    

    
    
    for case in cases:
        print('*************', case)
        
        dataset = MotionDataset(partition='test', max_samples=600, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame, num_past=args.num_past, case=case)

        ### choose the sixth tarjectory        
        x = torch.FloatTensor(dataset.X[6])
        v = torch.FloatTensor(dataset.V[6])

        n_samples, n_nodes=x.shape[:2]


        pivot = np.random.randint(args.num_past * args.delta_frame, n_samples-args.delta_frame, 1)[0]

        print('pivot: ', pivot)
        past_idx = [pivot-i*args.delta_frame for i in range(args.num_past, 0, -1)]


        loc = x[past_idx].to(device)
        vel = v[past_idx].to(device)
        
        edge_attr, charges = dataset.edge_attr, dataset.mole_idx.unsqueeze(-1)
        edge_attr, charges = edge_attr.to(device), charges.to(device)

        
        edges = dataset.get_edges(1, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]
        

        for model_name in tqdm(model_list):
            print('*********', model_name)
        
            exp_path = f"logs/motion_logs/exp_{case}/{model_name}"
            model=torch.load(f"{exp_path}/saved_model.pth", map_location=device)
            model.eval()

            loc_temp = loc.clone()
            vel_temp = vel.clone()

            with torch.no_grad():
                if model_name == 'egnn':
                    loc_pred = model(charges, loc_temp, vel_temp, edges, edge_attr)
                elif model_name == 'stag_egnn':
                    loc_pred = model(charges, loc_temp, vel_temp, edges, edge_attr)

                next_idx = pivot
                loc_end = x[next_idx].to(device)
                loss = loss_mse(loc_pred, loc_end)


                print(loss.item())
                print(loc_pred)



if __name__ == "__main__":
    main()




