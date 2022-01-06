from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from graph_conv import GraphConv

class Encoder(nn.Module):
    def __init__(self, dim_coor, in_channels: int, out_dim, layers_num, model_dim,out_channels_1,dropout):
        super(Encoder, self).__init__()
        self.layers_num=layers_num
        
        self.conv_layers = [GraphConv(coors=dim_coor,
                                      out_channels_1=out_channels_1,
                                      out_features=model_dim,
                                      label_dim=in_channels,
                                      dropout=dropout)] + \
                           [GraphConv(coors=dim_coor,
                                      out_channels_1=out_channels_1,
                                      out_features=model_dim,
                                      label_dim=model_dim,
                                      dropout=dropout) for _ in range(layers_num - 1)]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.fc1 = torch.nn.Linear(model_dim, out_dim)

  
    def forward(self, x, edge_index, pos):
        for i in range(self.layers_num):
            pos = torch.tensor(pos, dtype=torch.float32).cuda()
            #pos = pos.clone().detach()
            x = self.conv_layers[i](x, pos, edge_index)

        
        #x = global_mean_pool(data.x, data.batch)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)
    


class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos:torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index, pos)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
