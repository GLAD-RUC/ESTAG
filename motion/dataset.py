import numpy as np
import torch
import pickle as pkl
import os


class MotionDataset():
    """
    Motion Dataset

    """

    def __init__(self, partition, max_samples, delta_frame, data_dir, num_past, case='walk'):
        try:
            with open(os.path.join(data_dir, f'motion_{case}.pkl'), 'rb') as f:
                edges, X = pkl.load(f)
        except:
            raise RuntimeError('Unknown case')


        self.X = X
        for i in range(len(X)):
            ### (n_frame, n_nodes, 3)
            X[i] = X[i][:-1]
            X[i] = X[i] - np.mean(X[i])
        

        N = X[0].shape[1]
        if case == 'walk':
            train_case_id = [20, 1, 17, 13, 14, 9, 4, 2, 7, 5, 16]
            val_case_id = [3, 8, 11, 12, 15, 18]
            test_case_id = [6, 19, 21, 0, 22, 10]
        elif case == 'run':
            train_case_id = [1, 2, 5, 6, 10]
            val_case_id = [0, 4, 9, 8]
            test_case_id = [3, 7]
        elif case == 'basketball':
            train_case_id = [20, 1, 17, 13, 14, 9, 4, 2, 7, 5, 16, 18]
            val_case_id = [3, 0, 11, 21, 15, 23]
            test_case_id = [6, 19, 12, 8, 22, 10]
        else:
            raise RuntimeError('Unknown case')

        split_dir = os.path.join(data_dir, f'split_{case}.pkl')
        self.partition = partition

        try:
            with open(split_dir, 'rb') as f:
                print('Got Split!')
                split = pkl.load(f)
        except:
            np.random.seed(100)

              # sample 100 for each case
            if case  == 'walk':
                itv = 300
            elif case == 'run':
                itv = 70
            elif case == 'basketball':
                itv = 120
            else:
                raise RuntimeError('Unknown case')

            train_mapping = {}
            for i in train_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=50 if case == 'run' else 100, replace=False)
                train_mapping[i] = sampled
            val_mapping = {}
            for i in val_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=50 if case == 'run' else 100, replace=False)
                val_mapping[i] = sampled
            test_mapping = {}
            for i in test_case_id:
                # cur_x = X[i][:itv]
                sampled = np.random.choice(np.arange(itv), size=50 if case == 'run' else 100, replace=False)
                test_mapping[i] = sampled

            with open(split_dir, 'wb') as f:
                pkl.dump((train_mapping, val_mapping, test_mapping), f)

            print('Generate and save split!')
            split = (train_mapping, val_mapping, test_mapping)


        if partition == 'train':
            mapping = split[0]
        elif partition == 'val':
            mapping = split[1]
        elif partition == 'test':
            mapping = split[2]
        else:
            raise NotImplementedError()

        self.mapping = mapping
        each_len = max_samples // len(mapping)


        x_0, x_t = [], []
        for i in mapping:
            st = mapping[i][:each_len]
            #cur_x_0 = X[i][st]
            cur_x_0 = np.stack([X[i][[k+j*delta_frame for j in range(num_past)]] for k in st])

            #cur_x_t = X[i][st + delta_frame]
            cur_x_t = X[i][st + delta_frame*num_past]

            x_0.append(cur_x_0)
            x_t.append(cur_x_t)
        x_0 = np.concatenate(x_0, axis=0)
        x_t = np.concatenate(x_t, axis=0)

        print('Got {:d} samples!'.format(x_0.shape[0]))

        self.n_node = N

        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if self.atom_edge[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                        assert not self.atom_edge2[i][j]
                    if self.atom_edge2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([2])
                        assert not self.atom_edge[i][j]

        edges = [rows, cols]  # edges for equivariant message passing
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        self.edge_attr = edge_attr  # [edge, 3]
        self.edges = edges  # [2, edge]


        ### (n_samples, num_past, n_node, 3)
        self.x_0, self.x_t = torch.Tensor(x_0), torch.Tensor(x_t)


        mole_idx = np.ones(N)
        self.mole_idx = torch.Tensor(mole_idx)

        # self.cfg = self.sample_cfg()

        self.A=get_normalized_adj(self.atom_edge+self.atom_edge2)

    def sample_cfg(self):
        """
        Kinematics Decomposition
        """
        cfg = {}

        cfg['Stick'] = [(0, 11), (12, 13)]
        cfg['Stick'].extend([(2, 3), (7, 8), (17, 18), (24, 25)])

        cur_selected = []
        for _ in cfg['Stick']:
            cur_selected.append(_[0])
            cur_selected.append(_[1])

        cfg['Isolated'] = [[_] for _ in range(self.n_node) if _ not in cur_selected]
        if len(cfg['Isolated']) == 0:
            cfg.pop('Isolated')

        return cfg

    def __getitem__(self, i):

        # cfg = self.cfg

        # edge_attr = self.edge_attr
        # stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        # edges = self.edges

        # for m in range(len(edges[0])):
        #     row, col = edges[0][m], edges[1][m]
        #     if 'Stick' in cfg:
        #         for stick in cfg['Stick']:
        #             if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
        #                 stick_ind[m] = 1
        #     if 'Hinge' in cfg:
        #         for hinge in cfg['Hinge']:
        #             if (row, col) in [(hinge[0], hinge[1]), (hinge[1], hinge[0]), (hinge[0], hinge[2]), (hinge[2], hinge[0])]:
        #                 stick_ind[m] = 2
        # edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)  # [edge, 2]
        # cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}

        #return self.x_0[i], self.v_0[i], edge_attr, self.mole_idx.unsqueeze(-1), self.x_t[i], self.v_t[i], self.mole_idx.unsqueeze(-1), cfg
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

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg


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

if __name__ == '__main__':
    data = MotionDataset(partition='train', max_samples=500, delta_frame=30, data_dir='')