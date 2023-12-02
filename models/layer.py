from torch import nn
import torch
import math
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(hidden_size, filter_size), 
            nn.ReLU(), 
            nn.Linear(filter_size, hidden_size),
        )

    def forward(self, x):
        return self.layer(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_size=6):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size

        att_size = hidden_size // head_size
        self.att_size = att_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size, bias=False)
    
    def forward(self, q, k, v):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size

        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q.mul_(self.scale)
        x = torch.matmul(q, k) 
        x = torch.softmax(x, dim=3)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        return x

### Time attention
class TSALayer(nn.Module):
    def __init__(self, hidden_size, filter_size):
        super(TSALayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size)

    def forward(self, x):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        x = x + y

        return x


class AGL(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGL, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.linear = nn.Linear(node_num, node_num)

        
        self.out_liner = nn.Linear(cheb_k*dim_in, dim_out)

    def forward(self, x, node_embeddings):
        '''
            x: (b, n, c)
            node_embeddings: (n, d)
        '''
        node_num = node_embeddings.shape[0]

        ### (n, n)
        supports = F.softmax(F.relu(self.linear(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)))), dim=1)

        ### [(n, n), (n, n)]
        support_set = [torch.eye(node_num).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        
        ### (3, n, n)
        supports = torch.stack(support_set, dim=0)
    

        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3) 


        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  
        bias = torch.matmul(node_embeddings, self.bias_pool)  

        ####### fix here, replace weights and bias with liner layer
        ## (b, n, out_dim)
        # x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  
        x_gconv = self.out_liner(x_g.flatten(-2))

        return x_gconv



class AGLLayer(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGLLayer, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gconv_layer = AGL(node_num, dim_in, dim_out, cheb_k, embed_dim)

    def forward(self, x, node_embeddings):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        if x.shape[2] != self.node_num:
            x = x.view(batch_size, seq_len, self.node_num, -1)
        gconv_lst = []
        for t in range(seq_len):
            gconv_lst.append(self.gconv_layer(x[:, t, :, :], node_embeddings))
        
        ### (b, t, n*out_dim)
        output = torch.stack(gconv_lst, dim=1).view(batch_size, seq_len, -1)
        return output





class AGLTSA(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, hidden_size, filter_size, num_layers):
        super(AGLTSA, self).__init__()
        self.hidden_size = hidden_size
        
        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)

        self.register_buffer('inv_timescales', inv_timescales)
        

        ######### fix here, if dim_in != dim_out, it should be like this
        temp = [AGLLayer(node_num, dim_in, dim_out, cheb_k, embed_dim)]
        for _ in range(num_layers - 1): temp.append(AGLLayer(node_num, dim_out, dim_out, cheb_k, embed_dim))
        self.gconv_layers = nn.ModuleList(temp) 

        self.encoders = nn.ModuleList(
            [TSALayer(hidden_size, filter_size)
            for _ in range(num_layers)]
        )

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)

        ########## fix here, from (0, 0, 0, self.hidden_size % 2) -> (0, self.hidden_size % 2)
        signal = F.pad(signal, (0, self.hidden_size % 2))
        ## (1, n, n*out_dim)
        signal = signal.view(1, max_length, self.hidden_size)
        return signal

    def forward(self, inputs, node_embeddings):
        """
        inputs : [B, T, N, C]
        node_embeddings : [B, T, N, D]
        """
        # for encoding
        pos_enc = True
        encoder_output = inputs
        for gconv_layer, enc_layer in zip(self.gconv_layers, self.encoders):
            ### (b, t, n*out_dim)
            gconv_output = gconv_layer(encoder_output, node_embeddings)
            if pos_enc:
                gconv_output += self.get_position_encoding(gconv_output)
                pos_enc = False
            
            encoder_output = enc_layer(gconv_output)
        
        return self.last_norm(encoder_output)