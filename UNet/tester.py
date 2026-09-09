#%%

import torch
import numpy as np
import pandas as pd
import xarray as xr

from args import UNetParser
from dataset import UNetDataset
from unet import UnetGenerator_c
#%%

def UNettest(cfg, dataset, outdir, modelname):
    
    model = UnetGenerator_c(dataset.nx, dataset.ny,ngf= 16 )

    modelfile = outdir + modelname
    model.load_state_dict(torch.load(modelfile))
    model.to(cfg.device)
    model.eval()

    forcings = dataset.forcings
    pars = dataset.pars
    target = dataset.target

    files = np.arange(0, dataset.nfile)

    dLsm = np.ndarray([len(files), dataset.nt, 1, len(dataset.df.lat), len(dataset.df.lon)])
    vicout = np.ndarray([len(files), dataset.nt, 1, len(dataset.df.lat), len(dataset.df.lon)])

    flag = 0
    for file in files:
        for t in range(dataset.nt): 
            if t == 0:
                input = np.concatenate([forcings[0, t, :, :, :], pars[file, t, :, :, :], target[file, t, :, :, :]], axis=0)
            else:
                input = np.concatenate([forcings[0, t, :, :, :], pars[file, t, :, :, :], y_last], axis=0)

            # input = np.concatenate([forcings[0, t, :, :, :], pars[file, t, :, :, :], target[file, t, :, :, :]], axis=0)

            input = input[np.newaxis, :]   # the first dimension needs to be batch size
            input = torch.as_tensor(input, dtype=torch.float32)
            input = input.to(cfg.device)

            out = model(input)
            dLsm[flag, t, :, :, :] = out.detach().cpu().numpy()[0, :]
            vicout[flag, t, :, :, :] = target[file, t+1, :, :, :]

            y_last = dLsm[flag, t, :, :, :]

        flag = flag + 1

    dLsm = dLsm * dataset.stat['OUT_SOIL_MOIST_std'] + dataset.stat['OUT_SOIL_MOIST_mean']
    vicout = vicout * dataset.stat['OUT_SOIL_MOIST_std'] + dataset.stat['OUT_SOIL_MOIST_mean']

    # dLsm = dLsm * dataset.stat['OUT_EVAP_std'] + dataset.stat['OUT_EVAP_mean']
    # vicout = vicout * dataset.stat['OUT_EVAP_std'] + dataset.stat['OUT_EVAP_mean']

    
    dLsm = np.reshape(dLsm, [len(files), dataset.nt, len(dataset.df.lat), len(dataset.df.lon)])
    vicsm = np.reshape(vicout, [len(files), dataset.nt, len(dataset.df.lat), len(dataset.df.lon)])


    return dLsm, vicsm

#%%

if __name__ == '__main__':

    # cfg = LSTMParser().parse()
    # cfg = convlstmParser().parse()
    cfg = UNetParser().parse()


    trange = ['2015-01', '2015-12-26']
    fileidx = [1400, 1500]
    fvarLst = ['prcp', 'tas', 'dswrf', 'dlwrf', 'pres', 'wind', 'vp']
    pvarLst = ['infilt', 'Ds', 'Dsmax', 'Ws', 'expt', 'd2', 'd3']
    ovarLst = ['OUT_SOIL_MOIST']

    # testset = LSTMDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx, trange, is_train=False)
    # testset = convlstmDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx, trange, is_train=False)
    # testset = ARDenseNetDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx, trange, is_train=False)
    testset = UNetDataset(cfg, fvarLst, pvarLst, ovarLst, fileidx, trange, is_train=False)
    modelname = 'checkpoint.pt'
    seq_len = 30

    # outdir = '/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/lstm/2yr_100_batch400/'
    outdir = cfg.out_model_dir
    # outdir = '/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/ARnet/2yr_1200_batch200/'

    # dLsm, vicsm = LSTMtest(cfg, testset, outdir, modelname, seq_len)
    # dLsm, vicsm = ConvLSTMtest(cfg, testset, outdir, modelname, seq_len)
    # dLsm, vicsm = ARDensetest(cfg, testset, outdir, modelname)
    dLsm, vicsm = UNettest(cfg, testset, outdir, modelname)

    print(dLsm.shape)
#%%