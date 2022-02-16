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

from transformers import BertModel, BertTokenizer
import re

from tfrecord.torch.dataset import TFRecordDataset

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

step = 0
writer = SummaryWriter()
datasetTrain = ("PDB_DeepFri_train")
datasetTest = ("PDB_DeepFri_test") ###need to convert all from TFRecord #will be presplit into 80//10//10 for train//test//validate

datasetSeq = ("dataSEQ") ### will be all the sequences used in the PDB_DeepFri datastes

train_L = DataLoader(datasetTrain, batch_size = 60) #auto-structure data and def per/draw
test_L = DataLoader(datasetTest, batch_size = 60)
learning_rate = 0.001

class LSTM():
    def __init__(self):
        ###define dataset

    def Gen_Encode(self):#/# run Pretained protBert LSTM to generate encoding
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        ###for all seq in seq set = sequence_Example
            sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
            encoded_input = tokenizer(sequence_Example, return_tensors='pt')
            output = model(**encoded_input)
        ###put output into dataset =  output_set
        return output_set

class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(dataset.num_features, 128) ###needs a bit of testing and research
        self.pool1 = TopKPooling(128, ratio=2)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=2)


        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)  ###test scale down and const

    def forward(self, data):
        x, edge_index, batch = output_set, data.edge_index, data.batch ###take in output set here since it is now == feature matrix

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  ###syntax largly taken from PYG github for now but will need to be changed a bit later given new architecture
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)



###need to add the entire end and change GCN to additave method similar to what is used in DEEPFRI paper to work with everthing
###additionally a lot will have to be done in the creation of a custom dataset and in the integration of the LSTM this code is more ...
### a basic framework to give structure my plans and get some code written to its end

#### end will consist of new D.A.G tree graph of G.O terms so a lot of custom architecture and a dataset will be needed for that

































e
