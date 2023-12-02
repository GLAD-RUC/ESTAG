from traceback import print_tb
from turtle import forward
import torch
import numpy as np
import dgl
from torch import nn

from se3_dynamics.models import OurSE3Transformer, OursTFN
from se3_dynamics.utils.utils_profiling import * # load before other local modules


class OurDynamics(torch.nn.Module):
    def __init__(self, num_future, num_past, n_particles, n_dimesnion, device='cpu', nf=16, n_layers=3, act_fn=nn.ReLU(), model="se3_transformer", num_degrees=4, div=1):
        super().__init__()
        #self._transformation = transformation
        self._n_particles = n_particles
        self._n_dimension = n_dimesnion
        self._dim = self._n_particles * self._n_dimension

        if model == 'se3_transformer':
            self.se3 = OurSE3Transformer(num_layers=n_layers,
                                     num_channels=nf, edge_dim=0, act_fn=act_fn, num_degrees=num_degrees, div=div)
        elif model == 'tfn':
            self.se3 = OursTFN(num_layers=n_layers,
                                     num_channels=nf, edge_dim=0, div=1, act_fn=act_fn, num_degrees=num_degrees)
        else:
            raise Exception("Wrong model")

        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        self.graph = None
        self.to(device)
        self.reset_parameters()

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)


    def step(self, xs, vs, charges, edges):
        #n_batch = xs.shape[0]
        xs = xs.view(-1, self._n_particles, self._n_dimension)
        vs = vs.view(-1, self._n_particles, self._n_dimension)

        output = self.f(xs, vs, charges, edges).view(-1, self._n_dimension)

        return output

    def forward(self, xs, vs, charges, edges):
        x=torch.stack([self.step(xs[i], vs[i], charges, edges) for i in range(xs.shape[0])])
        x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat

    @profile
    def f(self, xs, vs, charges, edges):
        """
        :param xs:
        :return: xs_outputs.size() = xs.size()
        """
        # xs.size() --> (batch_size: 64, num_nodes: 4, dim: 3)

        # features = xs.new_ones((xs.size(0), xs.size(1), 1))
        # self.se3t()

        # 1. Transform xs to G
        if self.graph is None:
            self.graph = array_to_graph(xs, edges)
            self.graph.ndata['x'] = torch.zeros_like(self.graph.ndata['x'])
            self.graph.edata['d'] = torch.zeros_like(self.graph.edata['d'])

            indices_src, indices_dst =  edges  # [N, K]
            self.indices_src = indices_src
            self.indices_dst = indices_dst

        distance = xs[:, self.indices_dst] - xs[:, self.indices_src]

        #self.graph.ndata['x'] = xs.view(xs.size(0) * xs.size(1), 3)
        self.graph.ndata['vel'] = vs.view(xs.size(0) * vs.size(1), 3).unsqueeze(1)

        self.graph.ndata['f1'] = self.graph.ndata['vel']#torch.cat([self.graph.ndata['x'].unsqueeze(1), self.graph.ndata['vel']], dim=1)

        self.graph.ndata['f'] = charges.unsqueeze(2)
        # self.graph.ndata['f'] = charges
        self.graph.edata['d'] = distance.view(-1, 3)

        # G_gt = array_to_graph(xs)
        # print((self.graph.edata['d'] - G_gt.edata['d']).pow(2).sum())
        # assert False

        G = self.graph

        # 2. Transform G with se3t to G_out
        G_out = self.se3(G)

        # 3. Transform G_out to out

        out = G_out['1'].view(xs.size())

        # out = xs # TODO transform.


        return out + xs


@profile
def array_to_graph(xs, edges):
    B, N, D = xs.size()

    ### create graph
    # get u, v, and w

    # get neighbour indices here and use throughout entire network; this is a numpy function
    indices_src, indices_dst = edges  # [N, K]
    # indices_dst = indices_dst.flatten() # [N*K]
    # indices_src = np.repeat(np.array(range(N)), N-1)

    individual_graphs = []
    for b in range(B):
        example = xs[b]

        # example has shape [N, D=3]

        # Create graph (connections only, no bond or feature information yet)
        G = dgl.DGLGraph((indices_src, indices_dst), num_nodes=N)

        ### add bond & feature information to graph
        G.ndata['x'] = example  # node positions [N, ...]
        G.ndata['f'] = example.new_ones(size=[N, 1, 1])
        # G.ndata['f'] = torch.zeros_like(example)
        # G.ndata['f'] = f[...,None] # feature values [N, ...]
        # the following two lines will use the same ordering as specified in dgl.DGLGraph((u, v))
        # G.edata['w'] = w.astype(DTYPE) # bond information
        G.edata['d'] = example[indices_dst] - example[indices_src] # relative postion

        individual_graphs.append(G)

    batched_graph = dgl.batch(individual_graphs)

    # print(G.ndata['x'].size())
    # print(G.edata['d'].size())
    # print(batched_graph.ndata['x'].size())
    # print(batched_graph.edata['d'].size())
    #
    # print(batched_graph)
    # assert False

    return batched_graph
