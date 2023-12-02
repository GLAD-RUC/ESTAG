import torch
from torch import nn
# from models.gcl import *
from models.gcl import E_GCL_X, E_GCL_AT_X, GMNL, GCL
from einops import rearrange


class EGNN_X(nn.Module):
    def __init__(self,num_past, num_future,  in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.LeakyReLU(0.2), n_layers=4, coords_weight=3.0):
        super(EGNN_X, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_X(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        
        self.num_past=num_past
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)

        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)



    def forward(self, h, x, edges, edge_attr):
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        for i in range(0, self.n_layers):
            h, x = self._modules["gcl_%d" % i](h, x, edges, edge_attr=edge_attr, Fs=None)
        

        if x.shape[0]==1:
            x_hat=x.squeeze(0)
        else:
            x_hat = torch.einsum("ij,jkts->ikts", torch.softmax(self.theta,dim=1), x).squeeze(0)
        return x_hat



class ESTAG_X(nn.Module):
    def __init__(self, num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, fft, eat, device, \
                n_layers, n_nodes, nodes_att_dim=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0, with_mask=False, tempo=True):
        super(ESTAG_X, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes=n_nodes
        self.num_past = num_past
        self.tempo = tempo

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)

        for i in range(n_layers):
            self.add_module("egcl_%d" % (i*2+1), E_GCL_X(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT_X(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        
        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


        self.reset_parameters()
        self.to(self.device)
    
        

    def reset_parameters(self):
        #self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0


    def FFT(self, h, x, n_nodes, edges):
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        
        ### (b*n_node, 3, num_past)
        F = torch.fft.fftn(x_norm, dim=-1)

        ### (b*n_node, num_past-1)
        attn_val = self.attn_mlp(h[1:]).squeeze(-1).transpose(0, 1)

        F = F[..., 1:]
        F_i = F[edges[0]]
        F_j = F[edges[1]]


        ## (n_egde, num_past-1)
        edge_attr = torch.abs(torch.sum(torch.conj(F_i) * F_j, dim=-2))
        edge_attr = edge_attr * (attn_val[edges[0]] * attn_val[edges[1]])

        edge_attr_norm = edge_attr / (torch.sum(edge_attr, dim=-1, keepdim=True)+1e-6)

        ### (b*n_node, num_past-1)
        Fs = torch.abs(torch.sum(F**2, dim=-2))
        Fs = Fs * attn_val

        Fs_norm = Fs / (torch.sum(Fs, dim=-1, keepdim=True)+1e-6)

        # print(edge_attr_norm.shape)
        # print(Fs_norm.shape)
        # assert False
        return edge_attr_norm, Fs_norm


    def forward(self, h, x, edges, edge_attr):
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))

        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        Fs=None
        if self.fft:
            ### only CA
            edge_attr, Fs = self.FFT(h, x[:, :, 1, :], self.n_nodes, edges=edges)#only CA

            ### all node
            # edge_attr, Fs = self.FFT(x, self.n_nodes, edges=edges)

            ### using cached feature
            # edge_attr, Fs = edge_attr_fft, Fs_fft
        

        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)
                
            if self.eat:
                h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)
        
        if self.tempo:
            x_hat=torch.einsum("ij,jkts->ikts", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        else:
            x_hat = torch.einsum("ij,jkts->ikts", torch.softmax(self.theta,dim=1), x).squeeze(0)

        return x_hat



class GMN(nn.Module):
    def __init__(self,num_past, num_future,  in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.LeakyReLU(0.2), n_layers=4, coords_weight=3.0):
        super(GMN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.num_past = num_past

        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        for i in range(0, n_layers):
            self.add_module("gmnl_%d" % i, GMNL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)



    def forward(self, h, x, edges, edge_attr):
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        for i in range(0, self.n_layers):
            h, x = self._modules["gmnl_%d" % i](h, x, edges,edge_attr=edge_attr)
        
        if x.shape[0]==1:
            x_hat=x.squeeze(0)
        else:
            x_hat = torch.einsum("ij,jkts->ikts", torch.softmax(self.theta,dim=1), x).squeeze(0)
        return x_hat



class GNN_X(nn.Module):
    def __init__(self, num_past, num_future, input_dim, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN_X, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf = in_edge_nf ,
                                              act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, 4*3))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        self.num_past = num_past
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)

        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)



    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding

        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        
        x = self.decoder(h)
        x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat.reshape(-1,4,3)
