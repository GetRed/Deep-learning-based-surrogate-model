#%%
import sys
import os

project_dir = '/home/shengsy/sun2023/zrun/'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

print(f"Working dir: {os.getcwd()}")
print(f"Added path: {project_dir}")


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dense_ed import DenseED
from args import ARnetParser
from dataset import ARDenseNetDataset
from train import trainmodel

#%%
cfg = ARnetParserV2().parse()

trange_t = ['2009-01', '2010-12']
trange_v = ['2013-01', '2014-12']
fileidx_t = [0, 400]
fileidx_v = [1200, 1400]
fvarLst = ['prcp', 'tas', 'dswrf', 'dlwrf', 'pres', 'wind', 'vp']
pvarLst = ['infilt', 'Ds', 'Dsmax', 'Ws', 'expt', 'd2', 'd3']
ovarLst = ['OUT_SOIL_MOIST']

trainset = ARDenseNetDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx_t, trange_t)
validset = ARDenseNetDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx_v, trange_v, is_train=False)

train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True)
valid_loader = DataLoader(validset, batch_size=cfg.batch_size_valid, shuffle=True)


print(len(train_loader))
print(len(train_loader.dataset))

nx = len(fvarLst) + len(pvarLst) + 1
ny = len(ovarLst)

model = DenseED(nx, ny, blocks=cfg.blocks, growth_rate=cfg.growth_rate,
                drop_rate=cfg.drop_rate, bn_size=cfg.bn_size,
                num_init_features=cfg.init_features, bottleneck=cfg.bottleneck)
model.to(cfg.device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
loss_func = nn.MSELoss()

trainmodel_v2(model, optimizer, loss_func, train_loader, valid_loader, cfg)

#%%
