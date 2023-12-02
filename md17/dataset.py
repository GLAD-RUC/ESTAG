import numpy as np
import torch
import pickle as pkl
import os
import networkx as nx
from networkx.algorithms import tree



class MD17Dataset():
    """
    MD17 Dataset

    """

    def __init__(self, partition, max_samples, delta_frame, num_past, data_dir, molecule_type):
        # setup a split, tentative setting
        train_par, val_par, test_par = 0.1, 0.05, 0.05
        full_dir = os.path.join(data_dir, molecule_type + '_dft.npz')
        split_dir = os.path.join(data_dir, molecule_type + '_split.pkl')
        data = np.load(full_dir)
        self.partition = partition
        self.molecule_type = molecule_type

        ### (all_frame, n_node, 3)
        x = data['R']
        
        ### zero centered to avoid numerical instability
        if molecule_type == 'benzene':
            x_mean = x.mean(axis=-2, keepdims=True)
            x = x - x_mean
        # x_mean = x.mean(axis=(0,1), keepdims=True)
        # x = x - x_mean

        x = x[:-1]

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except:
            np.random.seed(100)

            _x = x[10000: -10000]

            train_idx = np.random.choice(np.arange(_x.shape[0]), size=int(train_par * _x.shape[0]), replace=False)
            flag = np.zeros(_x.shape[0])
            for _ in train_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            val_idx = np.random.choice(rest, size=int(val_par * _x.shape[0]), replace=False)
            for _ in val_idx:
                flag[_] = 1
            rest = [_ for _ in range(_x.shape[0]) if not flag[_]]
            test_idx = np.random.choice(rest, size=int(test_par * _x.shape[0]), replace=False)

            train_idx += 10000
            val_idx += 10000
            test_idx += 10000

            split = (train_idx, val_idx, test_idx)

            with open(split_dir, 'wb') as f:
                pkl.dump(split, f)
            print('Generate and save split!')

        if partition == 'train':
            st = split[0]
        elif partition == 'val':
            st = split[1]
        elif partition == 'test':
            st = split[2]
        else:
            raise NotImplementedError()

        st = st[:max_samples]

        z = data['z']
        x = x[:, z > 1, ...]
        z = z[z > 1]

        self.x = x

        
        x_0 = np.stack([x[[k+i*delta_frame for i in range(num_past)]] for k in st])
        x_t = x[st + delta_frame*num_past]


        print('Got {:d} samples!'.format(x_0.shape[0]))

        mole_idx = z
        n_node = mole_idx.shape[0]
        self.n_node = n_node

        _lambda = 1.6

        def d(_i, _j, _t):
            return np.sqrt(np.sum((x[_t][_i] - x[_t][_j]) ** 2))

        n = z.shape[0]

        self.Z = torch.Tensor(z)

        atom_edges = torch.ones(n, n).int()
        atom_edges = torch.zeros(n, n).int()
        for i in range(n):
            for j in range(n):
                if i != j:
                    _d = d(i, j, 0)
                    if _d < _lambda:
                        atom_edges[i][j] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_node):
            for j in range(n_node):
                if i != j:
                    '''if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j],d(i,j,0)])'''

                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 1.])
                        assert not self.atom_edge2[i][j]
                    if self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([mole_idx[i], mole_idx[j], 2.])
                        assert not self.atom_edge[i][j]

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = edges  # [2, edge]

        all_edges = {}

        for i in range(n):
            for j in range(i + 1, n):
                _d = d(i, j, 0)
                if _d < _lambda:
                    idx_i, idx_j = z[i], z[j]
                    if idx_i < idx_j:
                        idx_i, idx_j = idx_j, idx_i
                    if (idx_i, idx_j) in all_edges:
                        all_edges[(idx_i, idx_j)].append([i, j])
                    else:
                        all_edges[(idx_i, idx_j)] = [[i, j]]

        print(all_edges)
        # select the type of bonds to preserve the bond constraint
        conf_edges = []
        for key in all_edges:
            # if True:
            assert abs(key[0] - key[1]) <= 2
            conf_edges.extend(all_edges[key])

        print(conf_edges)
        self.conf_edges = conf_edges
        self.x_0, self.x_t = torch.Tensor(x_0), torch.Tensor(x_t)

        
        self.mole_idx = torch.Tensor(mole_idx)

        self.A=get_normalized_adj(self.atom_edge+self.atom_edge2)


    def __getitem__(self, i):

        return self.x_0[i], self.edge_attr, self.mole_idx.unsqueeze(-1), self.x_t[i]

    def __len__(self):
        return len(self.x_0)

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    #A = A + torch.diag(torch.ones(A.shape[0], dtype=torch.float32))
    A_ = torch.tensor(A, dtype=torch.float32)
    D = torch.sum(A_, axis=1)
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = torch.reciprocal(torch.sqrt(D))
    A_wave = torch.multiply(torch.multiply(diag.reshape((-1, 1)), A_),
                         diag.reshape((1, -1)))
    return A_wave