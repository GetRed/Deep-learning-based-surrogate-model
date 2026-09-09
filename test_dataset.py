#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import pickle
#%%
class FSTRDataset(Dataset):
    def __init__(self, cfg, fvarLst, pvarLst, ovarLst, prevLst, fileidx, tRange, is_train=True):
        super(FSTRDataset, self).__init__()

        # 添加以下三行
        self.fvarLst = fvarLst   # 动态变量名列表
        self.pvarLst = pvarLst   # 静态参数名列表
        self.ovarLst = ovarLst   # 目标变量名列表
        self.prevLst = prevLst          # 新增：前序变量列表
        
        self.dataPath = cfg.input_path
        self.seq_length = cfg.frames_input  # 读取模型输入序列长度（ConvLSTM的“时间帧数量”）
        self.df = xr.open_dataset(cfg.input_path + 'forcings_day.nc') #强迫数据
        self.dp = xr.open_dataset(cfg.input_path + 'par2000.nc') #静态参数
        self.do = xr.open_dataset(cfg.input_path + 'sm2000.nc') #目标变量
        self.nfile = fileidx[1]-fileidx[0] #文件总数
        self.fileidx = fileidx #文件范围
        self.tRange = tRange #时间范围
        self.nt = len(self.do.time.loc[tRange[0]:tRange[1]]) #有效时间帧
        self.is_train = is_train #训练标记

        if self.is_train:
            self.stat = {}
        else:
            scaler_file = cfg.out_dir + "train_data_scaler.bin"
            with open(scaler_file, mode='rb') as fp:
                self.stat = pickle.load(fp)

        self.forcings = self.getDataTs(self.df, fvarLst)
        self.pars = self.getDataConst(self.dp, pvarLst)
        self.target = self.getDataTs(self.do, ovarLst)
        self.prev = self.getDataTs(self.do, prevLst)

        if self.is_train:  # 训练集：保存统计量到文件（供测试集使用）
            file_path = cfg.out_dir + "train_data_scaler.bin"
            with open(file_path, mode='wb') as fp:
                pickle.dump(self.stat, fp)
        #  构建样本索引表（ConvLSTM核心适配点！）
        lookup = [(i, k) for i in range(self.nfile) for k in range(self.seq_length, self.nt)]
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}

        
    def __len__(self):
        return len(self.lookup_table) #样本总数
    

    def __getitem__(self, idx):
        file, indices = self.lookup_table[idx]

    # ✅ 正确：按 file 取 forcing
        input = np.concatenate([
            self.forcings[file, indices - self.seq_length + 1 : indices + 1, :, :, :],
            self.pars[file, indices - self.seq_length + 1 : indices + 1, :, :, :]
        ], axis=1)

        output = self.target[file, indices - self.seq_length + 1 : indices + 1, :, :, :]

        prevs = self.prev[file, indices - self.seq_length : indices, :, :, :]

        input = torch.as_tensor(input, dtype=torch.float32)
        output = torch.as_tensor(output, dtype=torch.float32)
        prevs = torch.as_tensor(prevs, dtype=torch.float32)

        return input, output, prevs


    def getDataTs(self, ds, varLst):
        # 通过检查第一个变量的维度来判断数据是否包含 file 维度
        first_var = ds[varLst[0]]
        has_file_dim = 'nfiles' in first_var.dims

        nvar = len(varLst) #变量个数
        nt = self.nt #有效步长

        # 始终分配 self.nfile 大小以保持 __getitem__ 索引兼容
        nfile = self.nfile
        data = np.ndarray([nfile, nt, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):
            var = ds[varLst[k]]
            if has_file_dim:
                dataTemp = var.loc[self.fileidx[0]:self.fileidx[1],self.tRange[0]:self.tRange[1],0,:,:].values  ## only use the first layer of output data
            else:
                dataTemp = var.loc[self.tRange[0]:self.tRange[1],:,:].values

            if self.is_train: #归一化
                mean = np.nanmean(dataTemp)
                std = np.nanstd(dataTemp)

                self.stat[varLst[k]+'_mean'] = mean
                self.stat[varLst[k]+'_std'] = std
            else:
                mean = self.stat[varLst[k]+'_mean']
                std = self.stat[varLst[k]+'_std']

            dataTemp = (dataTemp-mean)/std

            if has_file_dim:
                data[:, :, k, :, :] = dataTemp
            else:
                # 气象强迫数据没有 file 维度，广播到所有文件
                for f in range(nfile):
                    data[f, :, k, :, :] = dataTemp

        return data   # 返回处理后的五维时空序列数据（[nfile, nt, nvar, nlat, nlon]）

    def getDataConst(self, ds, varLst):

        nvar = len(varLst)
        nfile = self.nfile
        nt = self.nt

        data = np.ndarray([nfile, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):  #特殊变量的层筛选         
            if varLst[k] == 'expt':
                dataTemp = ds[varLst[k]].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1],1,:,:]  # second layer
            elif varLst[k] == 'd2':
                dataTemp = ds['depth'].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1],1,:,:]  # second layer
            elif varLst[k] == 'd3':
                dataTemp = ds['depth'].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1],2,:,:]  # third layer
            else:
                dataTemp = ds[varLst[k]].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1],:,:]

            if self.is_train:
                mean = np.nanmean(dataTemp)
                std = np.nanstd(dataTemp)

                self.stat[varLst[k]+'_mean'] = mean
                self.stat[varLst[k]+'_std'] = std
            else:
                mean = self.stat[varLst[k]+'_mean']
                std = self.stat[varLst[k]+'_std']

            dataTemp = (dataTemp-mean)/std
            
            data[:, k, :, :] = dataTemp
        out = np.repeat(np.reshape(data, [nfile, 1, nvar, len(ds.lat), len(ds.lon)]), nt, axis=1) #扩维

        return out #（[nfile, nt, nvar, nlat, nlon]）

# %%
