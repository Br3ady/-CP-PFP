import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import random
import numpy
import pdb
import sys
import os.path as osp

from torch_geometric.data import data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GATConv
import torch.optim as optim
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm





root = ('/tmp/ENZYMES')
step = 0
writer = SummaryWriter(("runs/PFP/1.3")) #def tensorboard storage path
dataset = TUDataset(root, 'ENZYMES') #def dataset (nodes represent 2nd struct elemnts 6 fundemental EC classes)

dataset = dataset.shuffle() #prep data
split = len(dataset) // 10 #split into training and test sets
train_S = dataset[split:]
test_S = dataset[:split]


train_L = DataLoader(train_S, batch_size = 36) #auto-structure data and def batch_size (optimized)
test_L = DataLoader(test_S, batch_size = 36)
learning_rate = 0.001 #optimized to fit CPU restrictions

    #for learning_rate in LR_L:

class GCN(nn.Module): #/#
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, 128) #GAT for better attention
        self.pool1 = TopKPooling(128, ratio=.6)
        self.conv2 = GATConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=.6)
        self.conv3 = GATConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=.6)

        self.lin1 = nn.Linear(256, 128) #lin for down scaling to classification values
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, dataset.num_classes) #scale down to classification size (6)




    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))#pass in node_feat matrix and adj matrix
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3 #sum pool

        x = F.relu(self.lin1(x))  ##run through linear layers and apply activations
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x



##Optimization
device = torch.device('cpu') #no cuda :(
model = GCN().to(device) #def gcn as in device
optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=5e-4)



def train(epoch):
    model.train()

    running_loss = 0.0
    for step, data in enumerate(train_L): #step for T.B
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data) #take output from forward
        loss = F.nll_loss(out, data.y) #Compare  output and target in NLL
        loss.backward() #backprop (~chage params)
        running_loss += data.num_graphs * loss.item() #calc loss
        optimizer.step() #step everything
    return running_loss / len(train_S) #return avg loss of training

def test(loader): #/ test
    model.eval()
    correct = 0

    for data in test_L:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in tqdm(range(300)): #of runs of full dataset
    #acc_list=[]
    #loss_list=[]

    loss = train(epoch)
    train_acc = test(train_L)
    test_acc = test(test_L)

    writer.add_scalar('Training loss', loss, global_step=step)
    writer.add_scalar('Test Accuracy', test_acc, global_step=step)
    writer.add_scalar('Train Accuracy', train_acc, global_step=step)

    step+=1
