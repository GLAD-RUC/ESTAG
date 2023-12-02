import argparse
import time
import torch
import torch.utils.data
from mdanalysis.dataset import MDAnalysisDataset, collate_mda

from models.model import AGLSTAN, STAG, STGCN
from models.model_x import *
import os
from torch import nn, optim
import json

import random
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='ESTAG')
parser.add_argument('--batch_size', type=int, default=100,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='logs/mdanalysis_logs', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--nf', type=int, default=16, metavar='N',
                    help='hidden dim')

parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='weight decay')
parser.add_argument('--data_dir', type=str, default='mdanalysis',
                    help='Data directory.')


parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')   
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--exp_name', type=str, default='exp_1000', metavar='N', help='experiment_name')
parser.add_argument('--num_past', type=int, default=10,
                    help='Number of length of whole past time series.')
parser.add_argument('--time_point', type=int, default=5,
                    help='Time point of past time series (egnn):1,5,10.')
parser.add_argument('--delta_frame', type=int, default=5,
                    help='Number of frames delta.')
parser.add_argument('--model', type=str, default='stag_gmn', metavar='N',
                    help='available models: baseline, egnn,stagmd')
parser.add_argument('--lr', type=float, default=5e-5, metavar='N',
                    help='learning rate')
parser.add_argument('--fft', type=eval, default=False,
                    help='Use FFT ')
parser.add_argument('--eat', type=eval, default=True,
                    help='Use EAT')   
parser.add_argument("--load_cached", action="store_true", help="Load cached dataset.")
parser.add_argument('--with_mask', action='store_true', default=False,
                    help='mask the future frame if use eat') 
parser.add_argument('--save_m', type=eval, default=True, help='whether to save model')  
parser.add_argument('--tempo', type=eval, default=True, help='Use temporal pooling') 




args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.cuda.set_device(0)
device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    exp_path = args.outf + f"/exp_{args.model}"
    # exp_path = args.outf + "/" + args.exp_name
    os.makedirs(exp_path)
except OSError:
    pass



def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

  
    dataset_train = MDAnalysisDataset('adk', partition='train', tmp_dir=args.data_dir,
                                      delta_frame=args.delta_frame, load_cached=args.load_cached, num_past=args.num_past)
    sampler = None
    shuffle = True
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=shuffle, sampler=sampler, drop_last=True,
                                               num_workers=0, collate_fn=collate_mda)

    
    dataset_val = MDAnalysisDataset('adk', partition='valid', tmp_dir=args.data_dir,
                                    delta_frame=args.delta_frame, load_cached=args.load_cached, num_past=args.num_past)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=shuffle,
                                             drop_last=True, num_workers=0, collate_fn=collate_mda)

    # Val and test do not need sampler.

    dataset_test = MDAnalysisDataset('adk', partition='test', tmp_dir=args.data_dir,
                                     delta_frame=args.delta_frame, load_cached=args.load_cached,
                                     test_rot=False, test_trans=False, num_past=args.num_past)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                              shuffle=shuffle,  drop_last=True,
                                              num_workers=0, collate_fn=collate_mda)



    #Adj=dataset_train.A.to(device)
    # start_time = time.time()
    # print("----Train (baseline)----:")
    # res = {'loss1': 0, 'loss5': 0, 'loss10': 0,  'counter': 0}
    # for batch_idx, data in enumerate(loader_train):
    #     loc,_, _ , _ , loc_end = data

    #     res['loss1'] += loss_mse(loc[0,:,:,:],loc_end).item()*args.batch_size
    #     res['loss5'] += loss_mse(loc[4,:,:,:],loc_end).item()*args.batch_size
    #     res['loss10'] += loss_mse(loc[9,:,:,:],loc_end).item()*args.batch_size
    #     res['counter'] += args.batch_size
    
    # print("Point 1: %.6f"%(res['loss1'] / res['counter']))
    # print("Point 5: %.6f"%(res['loss5'] / res['counter']))
    # print("Point 10: %.6f"%(res['loss10'] / res['counter']))

    #Point 1: 2.707518  Point 5: 2.408208  Point 10: 1.704617

    # start_time = time.time()
    # print("----Test (baseline)----:")
    # res = {'loss1': 0, 'loss5': 0, 'loss10': 0,  'counter': 0}
    # for batch_idx, data in enumerate(loader_test):
    #     loc, vel, edge_attr, charges, loc_end, edge_attr_fft, Fs_fft= data

    #     res['loss1'] += loss_mse(loc[0,:,:,:],loc_end).item()*args.batch_size
    #     res['loss5'] += loss_mse(loc[4,:,:,:],loc_end).item()*args.batch_size
    #     res['loss10'] += loss_mse(loc[9,:,:,:],loc_end).item()*args.batch_size
    #     res['counter'] += args.batch_size
    
    # print("Point 1: %.6f"%(res['loss1'] / res['counter']))
    # print("Point 5: %.6f"%(res['loss5'] / res['counter']))
    # print("Point 10: %.6f"%(res['loss10'] / res['counter']))

    # end_time = time.time()
    # print(end_time-start_time)
    # assert False

    #Point 1: 3.259700  Point 5: 3.301948  Point 10: 2.021766'''

    in_edge_nf = dataset_train.edge_attr.shape[-1]
    nodes_att_dim = 0
    if args.fft and args.model == 'estag':
        in_edge_nf = args.num_past-1
        nodes_att_dim = args.num_past-1    

    n_nodes = dataset_train.charges.shape[0]

    print("in_edge_nf",in_edge_nf)


    if args.model=='egnn':
        model = EGNN_X( num_past=args.num_past, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf, hidden_nf=args.nf, device=device, n_layers=args.n_layers)
    elif args.model=='estag':
        model = ESTAG_X(num_past=args.num_past, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf, hidden_nf=args.nf, fft=args.fft, eat=args.eat, device=device, n_layers=args.n_layers, n_nodes=n_nodes, with_mask=args.with_mask, tempo=args.tempo, nodes_att_dim=nodes_att_dim)
    elif args.model=='gmn':
        model = GMN( num_past=args.num_past, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf,hidden_nf=args.nf, device=device, n_layers=args.n_layers)
    elif args.model=='baseline':#past 1 --> future 1
        model = EGNN_X( num_past=1, num_future=1, in_node_nf=4, in_edge_nf=in_edge_nf,hidden_nf=args.nf, device=device, n_layers=args.n_layers)
    elif args.model=='stag_neq': #None-Equivariant STAG 
        out_dim = 4*3
        input_dim = 3*4+4
        model = STAG(num_nodes = n_nodes, num_features = input_dim, num_timesteps_input=args.num_past,num_timesteps_output=1, out_dim=out_dim).to(device)
    elif args.model=='gnn':
        input_dim = 3*4+4
        model = GNN_X(num_past=args.num_past, num_future=1, input_dim=input_dim, in_edge_nf=in_edge_nf, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    elif args.model=='stgcn':
        num_features = 3*4+4
        out_dim = 3*4
        model = STGCN(num_nodes=n_nodes, num_features=num_features, num_timesteps_input=args.num_past,num_timesteps_output=1, out_dim=out_dim, device=device)
    elif args.model == 'se3_transformer' or args.model == 'tfn':
        from se3_dynamics.dynamics import OurDynamics as SE3_Transformer
        model = SE3_Transformer(num_past=args.num_past, num_future=1, n_particles=n_nodes, n_dimesnion=3, nf=int(args.nf/args.degree), n_layers=args.n_layers, model=args.model, num_degrees=args.degree, div=1, device=device)
    elif args.model=='aglstan':
        num_features = 3*4+4
        out_dim = 3*4
        model = AGLSTAN(num_nodes=n_nodes, batch_size=args.batch_size, input_dim=num_features, output_dim=out_dim, window=args.num_past, num_layers=args.n_layers, filter_size=32, embed_dim=args.nf, cheb_k=3)
        model.to(device)
    else:
        raise Exception("Wrong model specified")

    print(model)



    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(0.2 * args.epochs), num_training_steps=args.epochs)


    results = {'epochs': [], 'test loss': [], 'val loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8

    #test_loss = train(model, optimizer, 5, loader_test, backprop=False)
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['val loss'].append(val_loss)
            results['test loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                if args.save_m:
                    torch.save(model, f'{exp_path}/saved_model.pth')
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best apoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))

        # scheduler.step()
        json_object = json.dumps(results, indent=4)
        with open(f"{exp_path}/loss.json", "w") as outfile:
            outfile.write(json_object)

    end_time = time.time()
    print(f'************training time(s) of {args.model}: {(end_time-start_time) / args.epochs}')
    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes = args.batch_size, data[0].shape[1]//args.batch_size
        data = [d.to(device) for d in data]

        loc, edge_attr, charges, loc_end, edge_attr_fft, Fs_fft= data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        #print(loss_mse(torch.mean(loc,axis=0),loc_end))
        optimizer.zero_grad()


        if args.model=='baseline':
            loc_pred=model(charges, loc[args.time_point-1].unsqueeze(0), edges, edge_attr)
        elif args.model=='stag_neq':
            feature = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)
            node = feature.permute(1,0,2).reshape(batch_size,n_nodes,feature.shape[0],feature.shape[2])
            Adj = loader.dataset.A.to(device)

            loc_pred = loc[-1] + model(Adj, node).reshape(loc.shape[1],4,3)
        elif args.model=='gmn':
            loc_pred=model(charges, loc, edges, edge_attr)
        elif args.model=='estag':
            loc_pred=model(charges, loc, edges, edge_attr)
        elif args.model=='egnn':
            loc_pred=model(charges, loc, edges, edge_attr)
        elif args.model=='gnn':
            nodes = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)
            loc_pred = model(nodes, edges, edge_attr)
        elif args.model=='stgcn':
            feature = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)

            node = feature.permute(1,0,2).reshape(batch_size,n_nodes,feature.shape[0],feature.shape[2])
            Adj = loader.dataset.A.to(device)

            loc_pred = loc[-1] + model(Adj, node).reshape(loc.shape[1],4,3)
            #loc_pred = model(Adj, node).reshape(loc.shape[1],4,3)
        elif args.model == 'aglstan':
            feature = torch.cat((charges.unsqueeze(0).repeat(loc.shape[0], 1, 1), loc.flatten(-2)), dim=-1)
            node = feature.permute(1,0,2).reshape(batch_size,feature.shape[0], n_nodes, feature.shape[2])

            loc_pred = model(node)
            # loc_pred = loc[-1] + loc_pred.reshape(loc.shape[1], 4, 3)
            loc_pred = loc_pred.reshape(loc.shape[1], 4, 3)
        else:
            raise Exception("Wrong model")


        loss = loss_mse(loc_pred, loc_end)

        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""

    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)
    with open(f"{exp_path}/loss.json") as f:
        loss=json.load(f)
    #plt.plot(loss['train loss'],label='Train')
    plt.plot(loss['epochs'],[np.mean(loss['train loss'][i*5:(i+1)*5]) for i in range(len(loss['train loss'])//5)],label='Train')
    plt.plot(loss['epochs'],loss['test loss'],label='Test')
    plt.legend()
    plt.title("Loss")
    plt.savefig(f'{exp_path}/loss.png')



