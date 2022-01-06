import numpy as np
import argparse
import os.path as osp
import random
import nni

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval_mydata_dice import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from DomainData_geo_newfeat import DomainData
from torch_geometric.data import Data
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='Wiki')
parser.add_argument('--param', type=str, default='local:wikics.json')
parser.add_argument('--seed', type=int, default=39788)
parser.add_argument('--verbose', type=str, default='train,eval,final')
parser.add_argument('--save_split', type=str, nargs='?')
parser.add_argument('--load_split', type=str, nargs='?')
parser.add_argument("--source1", type=str, default='data_source')

parser.add_argument("--target1", type=str, default='data_target')
default_param = {
    'learning_rate': 0.01,
    'num_hidden': 64,
    'num_proj_hidden': 32,
    'activation': 'prelu',
    'base_model': 'GCNConv',
    'num_layers': 3,
    'drop_edge_rate_1': 0.3,
    'drop_edge_rate_2': 0.4,
    'drop_feature_rate_1': 0.0,
    'drop_feature_rate_2': 0.0,
    'tau': 0.4,
    'num_epochs': 200,
    'weight_decay': 1e-5,
    'drop_scheme': 'degree',
}

# add hyper-parameters into parser
param_keys = default_param.keys()
for key in param_keys:
    parser.add_argument(f'--{key}', type=type(default_param[key]), default=(default_param[key]))
args = parser.parse_args(args=[])

# parse param
sp = SimpleParam(default=default_param)
param = sp(source=args.param, preprocess='nni')

# merge cli arguments and parsed param
for key in param_keys:
    if getattr(args, key) is not None:
        param[key] = getattr(args, key)

use_nni = args.param == 'nni'
if use_nni and args.device != 'cpu':
    args.device = 'cuda'

torch_seed = args.seed
torch.manual_seed(torch_seed)
random.seed(12345)

device = torch.device(args.device)

#path = osp.expanduser('~/datasets')
#path = osp.join(path, args.dataset)
#dataset = get_dataset(path, args.dataset)
dataset_pg=[]
dataset_s = DomainData("data/{}".format(args.source1), name=args.source1)
data_s = dataset_s[0]
dataset_pg.append(data_s)

dataset_t = DomainData("data/{}".format(args.target1), name=args.target1)
data_t = dataset_t[0]
dataset_pg.append(data_t)

encoder = Encoder(70, param['num_hidden'], get_activation(param['activation']),
                  base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=param['learning_rate'],
    weight_decay=param['weight_decay']
)

def train():
    model.train()
    

    
    loss_all=0
    for data in dataset_pg:
        data = data.to(device)
        optimizer.zero_grad()
        
        if param['drop_scheme'] == 'degree':
            drop_weights = degree_drop_weights(data.edge_index).to(device)
        elif param['drop_scheme'] == 'pr':
            drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        elif param['drop_scheme'] == 'evc':
            drop_weights = evc_drop_weights(data).to(device)
        else:
            drop_weights = None
        
        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1],len(data.y))
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        elif param['drop_scheme'] == 'pr':
            node_pr = compute_pr(data.edge_index,len(data.y))
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        elif param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(data)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = torch.ones((data.x.size(1),)).to(device)

        
        def drop_edge(idx: int):
            #global drop_weights
            
            if param['drop_scheme'] == 'uniform':
                return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
            elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
                return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
            else:
                raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')
            
        
        edge_index_1 = drop_edge(1)
        edge_index_2 = drop_edge(2)
        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

        if param['drop_scheme'] in ['pr', 'degree', 'evc']:
            x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
            x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    
    
        loss_all=loss_all+loss.item()
        loss.backward()
        optimizer.step()

    return loss_all/len(dataset_pg)

def test(dataset_pg, final=False):
    model.eval()
    
    z_data=[]
    for data in dataset_pg:
        data = data.to(device)
        z = model(data.x, data.edge_index)
        z=Data(z=z)
        z_data.append(z)
        

    evaluator = MulticlassEvaluator()
    
    acc = log_regression(z_data,dataset_pg, evaluator, num_epochs=1000)['acc']
    
    return acc

loss_acc = []
test_acc_array = []

for epoch in range(1, param['num_epochs'] + 1):
    loss = train()
    
    loss_acc.append(loss)
    #if 'train' in log:
    print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

    if epoch % 10 == 0:
        acc = test(dataset_pg)
        test_acc_array.append(acc)
        print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')
        
    plt.figure(1)
    #plt.plot(range(1,len(train_acc_array)+1), train_acc_array,'g',label='source_acc')
    plt.plot(range(1,len(test_acc_array)+1), test_acc_array,'r', label='target_acc')
    plt.legend()
    plt.savefig('test_acc.png')
    plt.close('all')
    
    plt.figure(2)
    #plt.plot(range(1,len(train_acc_array)+1), train_acc_array,'g',label='source_acc')
    plt.plot(range(1,len(loss_acc)+1), loss_acc,'r', label='loss')
    plt.legend()
    plt.savefig('train_loss.png')
    plt.close('all')



