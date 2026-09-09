#%%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resnet import ResnetGenerator_c
from args import ResNetParser
from dataset import ResNetDataset
from train import trainmodel
#%%

cfg = ResNetParser().parse()

trange_t = ['2009-01', '2010-12']
trange_v = ['2013-01', '2014-12']
fileidx_t = [0, 400]
fileidx_v = [1200, 1400]
fvarLst = ['prcp', 'tas', 'dswrf', 'dlwrf', 'pres', 'wind', 'vp']
pvarLst = ['infilt', 'Ds', 'Dsmax', 'Ws', 'expt', 'd2', 'd3']
# pvarLst = ['infilt', 'Ds', 'Dsmax', 'Ws', 'expt']
ovarLst = ['OUT_SOIL_MOIST']
# ovarLst = ['OUT_EVAP']

trainset = ResNetDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx_t, trange_t)
validset = ResNetDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx_v, trange_v, is_train=False)

train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True)
valid_loader = DataLoader(validset, batch_size=cfg.batch_size_valid, shuffle=True)

print(len(train_loader))  # number of batches
print(len(train_loader.dataset))  # number of total training samples

nx = len(fvarLst) + len(pvarLst) + 1
ny = len(ovarLst)

model = ResnetGenerator_c(nx, ny, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero')
# modelfile = '/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/ARnet/AE_7pnewin/checkpoint.pt'
# model.load_state_dict(torch.load(modelfile))
model.to(cfg.device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)   # optimize all cnn parameters
loss_func = nn.MSELoss()

trainmodel(model, optimizer, loss_func, train_loader, valid_loader, cfg)
#%%