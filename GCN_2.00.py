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


step = 0 ###move for batch
writer = SummaryWriter("runs/Prot/Comp(1.01+1.10)/")
dataset = ###Go + Prot_Seq
