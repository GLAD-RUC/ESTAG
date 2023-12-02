import os
import random
import numpy as np
from scipy.sparse import coo_matrix
import torch
#from pytorch3d import transforms
from torch.utils.data import Dataset

from MDAnalysisData import datasets
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import distances


class MDAnalysisDataset(Dataset):
    """
    NBodyDataset
    """
    def __init__(self, dataset_name, partition='train', tmp_dir=None, delta_frame=1, train_valid_test_ratio=None,
                 test_rot=False, test_trans=False, load_cached=False, cut_off=6, num_past=10):
        super().__init__()
        self.delta_frame = delta_frame
        self.dataset = dataset_name
        self.partition = partition
        self.load_cached = load_cached
        self.test_rot = test_rot
        self.test_trans = test_trans
        self.cut_off = cut_off
        self.num_past= num_past
        if load_cached:
            print(f'Loading {dataset_name} from cached data for {partition}...')
            tmp_dir = os.path.join(tmp_dir, 'adk_processed')
        self.tmp_dir = tmp_dir
        if train_valid_test_ratio is None:
            train_valid_test_ratio = [0.6, 0.2, 0.2]
        assert sum(train_valid_test_ratio) <= 1

        if load_cached:
            edges, self.edge_attr, self.charges, self.n_frames = torch.load(os.path.join(tmp_dir,
                                                                                         f'{dataset_name}.pkl'))
            self.edges = torch.stack(edges, dim=0)
            self.train_valid_test = [int(train_valid_test_ratio[0] * (self.n_frames - delta_frame)),
                                     int(sum(train_valid_test_ratio[:2]) * (self.n_frames - delta_frame))]
            return

        if dataset_name.lower() == 'adk':
            adk = datasets.fetch_adk_equilibrium(data_home=tmp_dir)
            #select
            
            self.data = mda.Universe(adk.topology, adk.trajectory)

            backbone = self.data.atoms.select_atoms("backbone")
            bb_by_res = backbone.split("residue")

            self.bb_by_res = [x for x in bb_by_res if len(x)==4]
            self.backbone = sum(self.bb_by_res)

            self.CA = self.backbone.select_atoms("name CA")
            self.id_ca = dict(zip(self.CA.ids,list(range(len(self.CA)))))
            #[y.resids[0] for y in self.bb_by_res]==[x.resid for x in self.CA]


        else:
            raise NotImplementedError(f'{dataset_name} is not available in MDAnalysisData.')

        # Local Graph information
        try:
            #self.charges = torch.tensor(self.data.atoms.charges)
            self.charges = torch.stack([torch.tensor(bb.charges) for bb in self.bb_by_res])
        except OSError:
            print(f'Charge error')
        try:
            #self.edges = torch.stack([torch.tensor(self.data.bonds.indices[:, 0], dtype=torch.long),
            #                          torch.tensor(self.data.bonds.indices[:, 1], dtype=torch.long)], dim=0)
            self.edges = torch.stack([torch.tensor(self.CA.bonds.indices[:, 0], dtype=torch.long),
                                      torch.tensor(self.CA.bonds.indices[:, 1], dtype=torch.long)], dim=0)
        except OSError:
            print(f'edges error')
        try:
            #self.edge_attr = torch.tensor([bond.length() for bond in self.data.bonds]).reshape(-1,1)
            self.edge_attr = torch.tensor([bond.length() for bond in self.CA.bonds]).reshape(-1,1)
        except OSError:
            print(f'edge_attr error')

        self.train_valid_test = [int(train_valid_test_ratio[0] * (len(self.data.trajectory) - self.delta_frame*self.num_past)),
                                 int(sum(train_valid_test_ratio[:2]) * (len(self.data.trajectory) - self.delta_frame*self.num_past))]
        

        x = torch.tensor(np.stack([self.data.trajectory[t].positions for t in range(len(self.data.trajectory))]))
        print(x.shape)
       
        x = x[:-1]

        self.X_bb = torch.stack([ x[:,bb.ids,:] for bb in self.bb_by_res ],axis=1)
        
        x_0 = np.ascontiguousarray(self.data.trajectory[0].positions)

        x_0_ca = x_0[self.CA.ids]


        edge_global = coo_matrix(distances.contact_matrix(x_0_ca,cutoff=self.cut_off, returntype="sparse"))
        edge_global.setdiag(False)
        edge_global.eliminate_zeros()
        self.edge_global = torch.stack([torch.tensor(edge_global.row, dtype=torch.long),
                                   torch.tensor(edge_global.col, dtype=torch.long)], dim=0)
        self.edge_global_attr = torch.norm(torch.tensor(x_0_ca)[self.edge_global[0], :] - torch.tensor(x_0_ca)[self.edge_global[1], :], p=2, dim=1).unsqueeze(-1)
        

        '''
        edge_attrs, Fss = [], []
        for i in range(self.X_bb.shape[0]-self.delta_frame*self.num_past):
            print(i)
            edge_attr, Fs = FFT(self.X_bb[[i+k*self.delta_frame for k in range(self.num_past)],:,1,:], len(self.CA),1, edges=self.edge_global)
            edge_attrs.append(edge_attr)
            Fss.append(Fs)
        edge_attrs_=torch.stack(edge_attrs)
        Fss_=torch.stack(Fss)
        torch.save(edge_attrs_,'mdanalysis/edge_attr_fft.pt')
        torch.save(Fss_,'mdanalysis/Fs_fft.pt')
        '''

        self.edge_attr_fft = torch.load('mdanalysis/edge_attr_fft.pt')
        self.Fs_fft = torch.load('mdanalysis/Fs_fft.pt')


        self.A = torch.zeros(self.charges.shape[0],self.charges.shape[0])
        for i in range(self.edge_global.shape[1]):
            self.A[self.edge_global[0,i], self.edge_global[1,i]] = self.edge_global_attr[i]

        self.A = get_normalized_adj(self.A)
        
        #self.A=get_normalized_adj(torch.ones(self.n_isolated,self.n_isolated))

    def __getitem__(self, i):
        charges, edge_attr = self.charges, self.edge_global_attr
        if len(charges.size()) == 1:
            charges = charges.unsqueeze(-1)
        if len(edge_attr.size()) == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        if self.partition == "valid":
            i = i + self.train_valid_test[0]
        elif self.partition == "test":
            i = i + self.train_valid_test[1]

        # Frames
        #num_past=10
        frame_0, frame_t = [i+k*self.delta_frame for k in range(self.num_past)], i + self.delta_frame*self.num_past


        return self.X_bb[frame_0], edge_attr, charges, self.X_bb[frame_t], self.edge_attr_fft[i], self.Fs_fft[i]

    def __len__(self):
        if self.load_cached:
            total_len = max(0, self.n_frames - self.delta_frame)
        else:
            total_len = max(0, len(self.data.trajectory) - self.delta_frame*self.num_past-1)
        if self.partition == 'train':
            return min(total_len, self.train_valid_test[0])
        if self.partition == 'valid':
            return max(0, min(total_len, self.train_valid_test[1]) - self.train_valid_test[0])
        if self.partition == 'test':
            return max(0, total_len - self.train_valid_test[1])

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edge_global[0]), torch.LongTensor(self.edge_global[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges


def collate_mda(data):
    #loc_0, edge_global, edge_global_attr, edges, edge_attr, charges, loc_t = zip(*data)

    loc_0, edge_attr, charges, loc_t, edge_attr_fft, Fs_fft= zip(*data)


    # edges
    #offset = torch.cumsum(torch.tensor([0] + [_.size(0) for _ in loc_0[0]], dtype=torch.long), dim=0)
    #edge_global = torch.cat(list(map(lambda _: _[0] + _[1], zip(edge_global, offset))), dim=-1)
    #edges = torch.cat(list(map(lambda _: _[0] + _[1], zip(edges, offset))), dim=-1)
    #edge_global_attr = torch.cat(edge_global_attr, dim=0).type(torch.float)

    #edges = torch.cat(edges,axis=1).type(torch.float)
    edge_attr = torch.cat(edge_attr, dim=0).type(torch.float)


    loc_0 = torch.cat(loc_0, axis=1).type(torch.float)
    loc_t = torch.cat(loc_t, axis=0).type(torch.float)

    charges = torch.cat(charges, dim=0).type(torch.float)

    edge_attr_fft = torch.cat(edge_attr_fft, dim=0).type(torch.float)
    Fs_fft = torch.cat(Fs_fft, dim=0).type(torch.float)


    return loc_0, edge_attr, charges, loc_t, edge_attr_fft, Fs_fft

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


def FFT(x, n_nodes, batch_size, edges):
    x_bar=torch.cat([torch.mean(x[:,i*n_nodes:(i+1)*n_nodes,:],axis=1).unsqueeze(1).repeat(1,n_nodes,1) for i in range(batch_size)],axis=1)
    x_norm=x-x_bar
    F=torch.stack([torch.fft.fft(x_norm[:,i,j]) for i in range(x_norm.shape[1]) for j in range(x_norm.shape[2])],axis=1).view(x.shape)
    #A=torch.stack([torch.stack([cal_similarity_fourier(F[j,n_nodes*i:n_nodes*(i+1),:]) for i in range(batch_size)]) for j in range(x.shape[0])],axis=-1)
    
    Fs=torch.abs(torch.einsum("ijt,ijt->ij",F,F))[1:,:].T
    Fs_norm=Fs/torch.sum(Fs,axis=1).unsqueeze(-1)

    #edge_attr=torch.stack([A[edges[0][i].item()//A.shape[1]][edges[0][i].item()%A.shape[1]][edges[1][i].item()%A.shape[1]] for i in range(len(edges[0]))])
    edge_attr = torch.stack([torch.abs(torch.sum(torch.conj(F[:,edges[0][i],:])*F[:,edges[1][i],:],axis=1)) for i in range(len(edges[0]))])
    edge_attr=edge_attr[:,1:]
    edge_attr_norm=edge_attr/torch.sum(edge_attr,axis=1).unsqueeze(-1)

    return edge_attr_norm, Fs_norm