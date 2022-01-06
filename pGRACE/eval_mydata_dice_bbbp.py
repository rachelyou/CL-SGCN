from typing import Optional

import torch
from torch.optim import Adam
import torch.nn as nn

import torch.nn.functional as F
from pGRACE.model import LogReg
from Dice_Loss import DiceLoss

def get_idx_split(dataset, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')


def log_regression(z_data,                   
                   train_dataset,
                   test_dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   #split: str = 'rand:0.1',
                   verbose: bool = False):
                   #preload_split=None):
    test_device = z_data[0].z.device if test_device is None else test_device
    ##z_t = z_data[-1].z.detach().to(test_device)
    num_hidden = z_data[0].z.size(1)
    #y = dataset_s.y.view(-1).to(test_device)
    ##y_t = dataset[-1].y.view(-1).to(test_device)
    num_classes = train_dataset[0].y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    #split = get_idx_split(dataset, split, preload_split)
    #split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()   ###
    dice_loss = DiceLoss().to(test_device) ####

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0
    #result_acc = []
    
    for epoch in range(num_epochs):
        classifier.train()
        
        for i in range(len(train_dataset)):
            z = z_data[i].z.detach().to(test_device)
            y = train_dataset[i].y.view(-1).to(test_device)
            optimizer.zero_grad()

        #output = classifier(z[split['train']])
            output = classifier(z)
            y_one_hot= F.one_hot(y, 2)
            #loss = nll_loss(f(output), y)  ####
            loss = dice_loss(output, y_one_hot) 
        #loss = nll_loss(f(output), y[split['train']])

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            #acc = evaluator.eval({
            #        'y_true': y_t.view(-1, 1),
            #        'y_pred': classifier(z_t).argmax(-1).view(-1, 1)
            #})['acc']
            for i in range(len(test_dataset)):
                z_t= z_data[len(train_dataset)+i].z.detach().to(test_device)
                y_t = test_dataset[i].y.view(-1).to(test_device)
                y_t1=F.one_hot(y_t,2)
                acc=1-dice_loss(classifier(z_t), y_t1)
                acc_loss=acc_loss+acc
                
            if (acc/len(test_dataset))>best_test_acc:
                best_test_acc=acc/len(test_dataset)
                best_epoch=epoch
                #result_acc = F.softmax(classifier(z_t),dim=1)
                #result_acc = torch.argmax(classifier(z_t),dim=1)
                #result_acc = classifier(z_t)
                #result_acc = F.one_hot(torch.argmax(classifier(z_t),dim = 1), 2)
            #if verbose:
    
    #np.savetxt('307_predict_ec.txt',result_acc,fmt='%d',delimiter=',')
    return {'acc': best_test_acc}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}

