import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import random
import numpy
import pdb
import sys

from torch_geometric.data import data
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv
import torch.optim as optim
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

step = 0
writer = SummaryWriter("runs/Prot/Comp(1.01+1.10)/") #def tensorboard storage path
dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS') #def dataset
##LR/BS optim only
#BS_L = [16, 32, 64, 128, 256]
#LR_L = [.01, .001, .0001, .003, .0003, .000001]

dataset = dataset.shuffle() #prep data
split = len(dataset) // 10
train_S = dataset[split:]
test_S = dataset[:split]


## loops for LR/BS optim only
#for batch_size in BS_L:
train_L = DataLoader(train_S, batch_size = 60) #auto-structure data and def per/draw
test_L = DataLoader(test_S, batch_size = 60)
learning_rate = 0.001 ###change to fit CPU and tests

    #for learning_rate in LR_L:

class GCN(nn.Module): #/#
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 128) ###dont condense bc no point rn
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)

        self.lin1 = nn.Linear(128, 64) #lin for classification
        self.lin2 = nn.Linear(64, dataset.num_classes) #scale down to classification size




    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        x = gap(x, batch) ###basic pool for classification on entire graph class (don't do for node-wide pred)


        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training) #prevent co-adaption
        x = F.relu(self.lin2(x))
        return F.log_softmax(x, dim=-1)




device = torch.device('cpu') ###need to run tests and better optim cpu for training tasks
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=5e-4)



def train(epoch):
    model.train()

    running_loss = 0.0
    for step, data in enumerate(train_L): #step for T.B
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data) #take output from forward
        loss = F.nll_loss(out, data.y) #pass out and predefined target
        loss.backward() #backprop
        running_loss += data.num_graphs * loss.item()
        optimizer.step()
    return running_loss / len(train_S)

def test(loader): #/ test
    model.eval()
    correct = 0

    for data in test_L:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in tqdm(range(20)):
    acc_list=[]
    loss_list=[]

    loss = train(epoch)
    train_acc = test(train_L)
    test_acc = test(test_L)

    #acc_list.append(train_acc)
    #loss_list.append(loss)

    writer.add_scalar('Training loss', loss, global_step=step)
    writer.add_scalar('Test Accuracy', test_acc, global_step=step)
    writer.add_scalar('Train Accuracy', train_acc, global_step=step)

#writer.add_hparams({'LR': learning_rate, 'BS': batch_size}, {'Acc' : sum(acc_list)/len(acc_list), 'Loss' : sum(loss_list)/len(loss_list)})
###^better ways to do this but not gonna spend time on them for this basic of an nn since ill be chaning a lot soon in...
##^implamenting better methods
    step+=1
