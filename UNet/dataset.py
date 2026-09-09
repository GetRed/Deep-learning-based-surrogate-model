#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import pickle 
import pandas as pd
#%%
class UNetDataset(Dataset):
    def __init__(self, cfg, fvarLst, pvarLst, ovarLst, fileidx, tRange, is_train=True):
        super(UNetDataset, self).__init__()
        self.dataPath = cfg.input_path
        #self.df = xr.open_dataset(cfg.input_path + 'forcings_short.nc')
        self.df = xr.open_dataset(cfg.input_path + 'forcings_day.nc')
        self.dp = xr.open_dataset(cfg.input_path + 'par2000.nc')
        self.do = xr.open_dataset(cfg.input_path + 'sm2000.nc')
        #self.do = xr.open_dataset(cfg.input_path + 'e2000.nc')

        self.nx = len(fvarLst) + len(pvarLst) + 1 # 输入维度：强迫变量数 + 参数变量数 + 1（初始状态变量）
        self.ny = len(ovarLst)  # 输出维度：目标变量数（这里是蒸散发变量数）(现在改成sm一层的了)
        self.nfile = fileidx[1]-fileidx[0]
        self.fileidx = fileidx #文件索引范围
        self.tRange = tRange
        self.nt = len(self.do.time.loc[tRange[0]:tRange[1]])
        self.is_train = is_train

        if self.is_train:
            self.stat = {}
        else:
            scaler_file = cfg.out_dir + "train_data_scaler.bin"
            with open(scaler_file, mode='rb') as fp:
                self.stat = pickle.load(fp)
        # （注释代码）备用：加载预训练的统计量（如AE模型的统计量，用于迁移学习）
        # with open('/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/ARnet/AE_7pnewin/train_data_scaler.bin', mode='rb') as fp:
        #     self.statini = pickle.load(fp)

        self.forcings = self.getDataTs(self.df, fvarLst)
        self.pars = self.getDataConst(self.dp, pvarLst)
        self.target = self.getDataTs(self.do, ovarLst) #后面有定义这三种处理函数
        
        #初始状态变量
        moist_ini = np.full((self.nfile, 1, 1, len(self.do.lat), len(self.do.lon)), 20)
        moist_ini = (moist_ini -self.stat['OUT_SOIL_MOIST_mean'])/self.stat['OUT_SOIL_MOIST_std']

        #moist_ini = np.full((self.nfile, 1, 1, len(self.do.lat), len(self.do.lon)), 0.5)
        #moist_ini = (moist_ini -self.stat['OUT_EVAP_mean'])/self.stat['OUT_EVAP_std'] #归一化

        self.target= np.concatenate([moist_ini, self.target], axis=1)  #拼接目标变量：将初始状态（1个时间步）拼接到目标变量的时间维度前

        if self.is_train: # 训练集：保存统计量到文件
            file_path = cfg.out_dir + "train_data_scaler.bin"
            with open(file_path, mode='wb') as fp:
                pickle.dump(self.stat, fp)

        lookup = [(i,k) for i in range(self.nfile)  for k in range(self.nt)]
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}  #构建样本索引表

        
    def __len__(self):
        return len(self.lookup_table)
    

    def __getitem__(self, idx):
        file, indices = self.lookup_table[idx]  # 从索引表获取（文件i，时间步k）
        input = np.concatenate([self.forcings[0, indices, :, :, :], self.pars[file, indices, :, :, :], self.target[file, indices, :, :, :] ], axis=0) #targets的 indices 是上一时刻，因为 self.target 包含了初始状态
        output = self.target[file, indices+1, :, :, :]

        input = torch.from_numpy(input).float()
        output = torch.from_numpy(output).float()

        return input, output


    def getDataTs(self, ds, varLst):

        if 'nfiles' not in ds.dims: #维度名判断文件数
            nfile = 1
        else:
            nfile = self.nfile

        nvar = len(varLst)
        nt = self.nt

        data = np.ndarray([nfile, nt, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):
            var = ds[varLst[k]]
            if len(var.dims) > 4:  # model output(soil moisture)
                dataTemp = var.loc[self.fileidx[0]:self.fileidx[1],self.tRange[0]:self.tRange[1],0,:,:].values  ## only use the first layer of output data
            #if len(var.dims) > 3:  # model output(ET)
            #    dataTemp = var.loc[self.fileidx[0]:self.fileidx[1],self.tRange[0]:self.tRange[1],:,:].values 
            else:
                dataTemp = var.loc[self.tRange[0]:self.tRange[1],:,:].values
            
            if self.is_train:
                mean = np.nanmean(dataTemp)
                std = np.nanstd(dataTemp)

                self.stat[varLst[k]+'_mean'] = mean
                self.stat[varLst[k]+'_std'] = std
            else:
                mean = self.stat[varLst[k]+'_mean']
                std = self.stat[varLst[k]+'_std']

            dataTemp = (dataTemp-mean)/std

            if len(var.dims) == 3: # forcing
                data[0, :, k, :, :] = dataTemp
            else:
                data[:, :, k, :, :] = dataTemp

       
        return data

    def getDataConst(self, ds, varLst):

        nvar = len(varLst)
        nfile = self.nfile
        nt = self.nt

        data = np.ndarray([nfile, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):           
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

            # dataTemp = (dataTemp - self.statini[varLst[k]+'_mean'])/self.statini[varLst[k]+'_std']
            
            data[:, k, :, :] = dataTemp
        out = np.repeat(np.reshape(data, [nfile, 1, nvar, len(ds.lat), len(ds.lon)]), nt, axis=1)

        return out
#%%

class ResNetDataset(Dataset):
    def __init__(self, cfg, fvarLst, pvarLst, ovarLst, fileidx, tRange, is_train=True):
        super(ResNetDataset, self).__init__()
        self.dataPath = cfg.input_path
        #self.df = xr.open_dataset(cfg.input_path + 'forcings_short.nc')
        self.df = xr.open_dataset(cfg.input_path + 'forcings_day.nc')
        self.dp = xr.open_dataset(cfg.input_path + 'par2000.nc')
        self.do = xr.open_dataset(cfg.input_path + 'sm2000.nc')
        #self.do = xr.open_dataset(cfg.input_path + 'e2000.nc')

        self.nx = len(fvarLst) + len(pvarLst) + 1 # 输入维度：强迫变量数 + 参数变量数 + 1（初始状态变量）
        self.ny = len(ovarLst)  # 输出维度：目标变量数（这里是蒸散发变量数）(现在改成sm一层的了)
        self.nfile = fileidx[1]-fileidx[0]
        self.fileidx = fileidx #文件索引范围
        self.tRange = tRange
        self.nt = len(self.do.time.loc[tRange[0]:tRange[1]])
        self.is_train = is_train

        if self.is_train:
            self.stat = {}
        else:
            scaler_file = cfg.out_dir + "train_data_scaler.bin"
            with open(scaler_file, mode='rb') as fp:
                self.stat = pickle.load(fp)
        # （注释代码）备用：加载预训练的统计量（如AE模型的统计量，用于迁移学习）
        # with open('/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/ARnet/AE_7pnewin/train_data_scaler.bin', mode='rb') as fp:
        #     self.statini = pickle.load(fp)

        self.forcings = self.getDataTs(self.df, fvarLst)
        self.pars = self.getDataConst(self.dp, pvarLst)
        self.target = self.getDataTs(self.do, ovarLst) #后面有定义这三种处理函数
        
        #初始状态变量
        moist_ini = np.full((self.nfile, 1, 1, len(self.do.lat), len(self.do.lon)), 20)
        moist_ini = (moist_ini -self.stat['OUT_SOIL_MOIST_mean'])/self.stat['OUT_SOIL_MOIST_std']

        #moist_ini = np.full((self.nfile, 1, 1, len(self.do.lat), len(self.do.lon)), 0.5)
        #moist_ini = (moist_ini -self.stat['OUT_EVAP_mean'])/self.stat['OUT_EVAP_std'] #归一化

        self.target= np.concatenate([moist_ini, self.target], axis=1)  #拼接目标变量：将初始状态（1个时间步）拼接到目标变量的时间维度前

        if self.is_train: # 训练集：保存统计量到文件
            file_path = cfg.out_dir + "train_data_scaler.bin"
            with open(file_path, mode='wb') as fp:
                pickle.dump(self.stat, fp)

        lookup = [(i,k) for i in range(self.nfile)  for k in range(self.nt)]
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}  #构建样本索引表

        
    def __len__(self):
        return len(self.lookup_table)
    

    def __getitem__(self, idx):
        file, indices = self.lookup_table[idx]  # 从索引表获取（文件i，时间步k）
        input = np.concatenate([self.forcings[0, indices, :, :, :], self.pars[file, indices, :, :, :], self.target[file, indices, :, :, :] ], axis=0) #targets的 indices 是上一时刻，因为 self.target 包含了初始状态
        output = self.target[file, indices+1, :, :, :]

        input = torch.from_numpy(input).float()
        output = torch.from_numpy(output).float()

        return input, output


    def getDataTs(self, ds, varLst):

        if 'nfiles' not in ds.dims: #维度名判断文件数
            nfile = 1
        else:
            nfile = self.nfile

        nvar = len(varLst)
        nt = self.nt

        data = np.ndarray([nfile, nt, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):
            var = ds[varLst[k]]
            if len(var.dims) > 4:  # model output(soil moisture)
                dataTemp = var.loc[self.fileidx[0]:self.fileidx[1],self.tRange[0]:self.tRange[1],0,:,:].values  ## only use the first layer of output data
            #if len(var.dims) > 3:  # model output(ET)
            #    dataTemp = var.loc[self.fileidx[0]:self.fileidx[1],self.tRange[0]:self.tRange[1],:,:].values 
            else:
                dataTemp = var.loc[self.tRange[0]:self.tRange[1],:,:].values
            
            if self.is_train:
                mean = np.nanmean(dataTemp)
                std = np.nanstd(dataTemp)

                self.stat[varLst[k]+'_mean'] = mean
                self.stat[varLst[k]+'_std'] = std
            else:
                mean = self.stat[varLst[k]+'_mean']
                std = self.stat[varLst[k]+'_std']

            dataTemp = (dataTemp-mean)/std

            if len(var.dims) == 3: # forcing
                data[0, :, k, :, :] = dataTemp
            else:
                data[:, :, k, :, :] = dataTemp

       
        return data

    def getDataConst(self, ds, varLst):

        nvar = len(varLst)
        nfile = self.nfile
        nt = self.nt

        data = np.ndarray([nfile, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):           
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

            # dataTemp = (dataTemp - self.statini[varLst[k]+'_mean'])/self.statini[varLst[k]+'_std']
            
            data[:, k, :, :] = dataTemp
        out = np.repeat(np.reshape(data, [nfile, 1, nvar, len(ds.lat), len(ds.lon)]), nt, axis=1)

        return out
# %%

class ARDenseNetDataset(Dataset):
    def __init__(self, cfg, fvarLst, pvarLst, ovarLst, fileidx, tRange, is_train=True):
        super(ARDenseNetDataset, self).__init__()
        self.dataPath = cfg.input_path
        #self.df = xr.open_dataset(cfg.input_path + 'forcings_short.nc')
        self.df = xr.open_dataset(cfg.input_path + 'forcings_day.nc')
        self.dp = xr.open_dataset(cfg.input_path + 'par2000.nc')
        self.do = xr.open_dataset(cfg.input_path + 'sm2000.nc')
        #self.do = xr.open_dataset(cfg.input_path + 'e2000.nc')

        self.nx = len(fvarLst) + len(pvarLst) + 1 # 输入维度：强迫变量数 + 参数变量数 + 1（初始状态变量）
        self.ny = len(ovarLst)  # 输出维度：目标变量数（这里是蒸散发变量数）(现在改成sm一层的了)
        self.nfile = fileidx[1]-fileidx[0]
        self.fileidx = fileidx #文件索引范围
        self.tRange = tRange
        self.nt = len(self.do.time.loc[tRange[0]:tRange[1]])
        self.is_train = is_train

        if self.is_train:
            self.stat = {}
        else:
            scaler_file = cfg.out_dir + "train_data_scaler.bin"
            with open(scaler_file, mode='rb') as fp:
                self.stat = pickle.load(fp)
        # （注释代码）备用：加载预训练的统计量（如AE模型的统计量，用于迁移学习）
        # with open('/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/ARnet/AE_7pnewin/train_data_scaler.bin', mode='rb') as fp:
        #     self.statini = pickle.load(fp)

        self.forcings = self.getDataTs(self.df, fvarLst)
        self.pars = self.getDataConst(self.dp, pvarLst)
        self.target = self.getDataTs(self.do, ovarLst) #后面有定义这三种处理函数
        
        #初始状态变量
        moist_ini = np.full((self.nfile, 1, 1, len(self.do.lat), len(self.do.lon)), 20)
        moist_ini = (moist_ini -self.stat['OUT_SOIL_MOIST_mean'])/self.stat['OUT_SOIL_MOIST_std']

        #moist_ini = np.full((self.nfile, 1, 1, len(self.do.lat), len(self.do.lon)), 0.5)
        #moist_ini = (moist_ini -self.stat['OUT_EVAP_mean'])/self.stat['OUT_EVAP_std'] #归一化

        self.target= np.concatenate([moist_ini, self.target], axis=1)  #拼接目标变量：将初始状态（1个时间步）拼接到目标变量的时间维度前

        if self.is_train: # 训练集：保存统计量到文件
            file_path = cfg.out_dir + "train_data_scaler.bin"
            with open(file_path, mode='wb') as fp:
                pickle.dump(self.stat, fp)

        lookup = [(i,k) for i in range(self.nfile)  for k in range(self.nt)]
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}  #构建样本索引表

        
    def __len__(self):
        return len(self.lookup_table)
    

    def __getitem__(self, idx):
        file, indices = self.lookup_table[idx]  # 从索引表获取（文件i，时间步k）
        input = np.concatenate([self.forcings[0, indices, :, :, :], self.pars[file, indices, :, :, :], self.target[file, indices, :, :, :] ], axis=0) #targets的 indices 是上一时刻，因为 self.target 包含了初始状态
        output = self.target[file, indices+1, :, :, :]

        input = torch.from_numpy(input).float()
        output = torch.from_numpy(output).float()

        return input, output


    def getDataTs(self, ds, varLst):

        if 'nfiles' not in ds.dims: #维度名判断文件数
            nfile = 1
        else:
            nfile = self.nfile

        nvar = len(varLst)
        nt = self.nt

        data = np.ndarray([nfile, nt, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):
            var = ds[varLst[k]]
            if len(var.dims) > 4:  # model output(soil moisture)
                dataTemp = var.loc[self.fileidx[0]:self.fileidx[1],self.tRange[0]:self.tRange[1],0,:,:].values  ## only use the first layer of output data
            #if len(var.dims) > 3:  # model output(ET)
            #    dataTemp = var.loc[self.fileidx[0]:self.fileidx[1],self.tRange[0]:self.tRange[1],:,:].values 
            else:
                dataTemp = var.loc[self.tRange[0]:self.tRange[1],:,:].values
            
            if self.is_train:
                mean = np.nanmean(dataTemp)
                std = np.nanstd(dataTemp)

                self.stat[varLst[k]+'_mean'] = mean
                self.stat[varLst[k]+'_std'] = std
            else:
                mean = self.stat[varLst[k]+'_mean']
                std = self.stat[varLst[k]+'_std']

            dataTemp = (dataTemp-mean)/std

            if len(var.dims) == 3: # forcing
                data[0, :, k, :, :] = dataTemp
            else:
                data[:, :, k, :, :] = dataTemp

       
        return data

    def getDataConst(self, ds, varLst):

        nvar = len(varLst)
        nfile = self.nfile
        nt = self.nt

        data = np.ndarray([nfile, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):           
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

            # dataTemp = (dataTemp - self.statini[varLst[k]+'_mean'])/self.statini[varLst[k]+'_std']
            
            data[:, k, :, :] = dataTemp
        out = np.repeat(np.reshape(data, [nfile, 1, nvar, len(ds.lat), len(ds.lon)]), nt, axis=1)

        return out

class FSTRDataset(Dataset):
    def __init__(self, cfg, fvarLst, pvarLst, ovarLst, fileidx, tRange, seq_len, is_train=True):
        super(FSTRDataset, self).__init__()

        self.fvarLst = fvarLst   # 动态变量名列表
        self.pvarLst = pvarLst   # 静态参数名列表
        self.ovarLst = ovarLst   # 目标变量名列表
        
        self.dataPath = cfg.input_path
        self.df = xr.open_dataset(cfg.input_path + 'forcings_day.nc')
        self.dp = xr.open_dataset(cfg.input_path + 'par2000.nc')
        self.do = xr.open_dataset(cfg.input_path + 'sm2000.nc')
        self.nfile = fileidx[1] - fileidx[0]
        self.fileidx = fileidx
        self.tRange = tRange
        self.is_train = is_train

        # 计算原始时间范围内的天数（要预测的天数）
        start = pd.Timestamp(tRange[0])
        end = pd.Timestamp(tRange[1])
        self.nt = len(pd.date_range(start, end))
        self.seq_len = seq_len

        # target 需要包含前一天
        init_date = (start - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        target_tRange = [init_date, tRange[1]]          # 长度 = nt + 1

        if self.is_train:
            self.stat = {}
        else:
            scaler_file = cfg.out_dir + "train_data_scaler.bin"
            with open(scaler_file, mode='rb') as fp:
                self.stat = pickle.load(fp)

        # 读取数据：forcings 和 pars 只用原始 tRange
        self.forcings = self.getDataTs(self.df, fvarLst, time_range=tRange)          # [nfile, nt, fvar_len, H, W]
        self.pars = self.getDataConst(self.dp, pvarLst, time_range=tRange)           # [nfile, nt, pvar_len, H, W]
        # target 使用扩展范围
        self.target = self.getDataTs(self.do, ovarLst, time_range=target_tRange)     # [nfile, nt+1, out_len, H, W]


        if self.is_train:
            file_path = cfg.out_dir + "train_data_scaler.bin"
            with open(file_path, mode='wb') as fp:
                pickle.dump(self.stat, fp)

        # 构建样本索引：offset 为起始预测日相对于 start 的偏移，取值范围 0 到 nt - seq_len
        lookup = [(i, offset) for i in range(self.nfile) for offset in range(0, self.nt - self.seq_len + 1)]
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}

    def __len__(self):
        return len(self.lookup_table)

    def __getitem__(self, idx):
        file, offset = self.lookup_table[idx]   # offset: 起始预测日相对于 start 的偏移

    # 前一时刻土壤湿度（初始条件）：第 start+offset-1 天的真实值，对应 target 索引 offset
        prev_sm = self.target[file, offset]          # shape: (out_len, H, W)

    # 连续 seq_len 个时刻的气象强迫：从 start+offset 到 start+offset+seq_len-1
        forcings = self.forcings[file, offset : offset+self.seq_len]   # shape: (seq_len, fvar_len, H, W)

    # 静态参数（任意时刻，取 offset 时刻）
        static_inputs = self.pars[file, offset]      # shape: (pvar_len, H, W)

    # 连续 seq_len 个目标值：从 start+offset 到 start+offset+seq_len-1
        target = self.target[file, offset+1 : offset+self.seq_len+1]   # shape: (seq_len, out_len, H, W)

    # 转换为 tensor
        prev_sm = torch.from_numpy(prev_sm).float().unsqueeze(0)          # (1, out_len, H, W)
        forcings = torch.from_numpy(forcings).float()                     # (seq_len, fvar_len, H, W)
        static_inputs = torch.from_numpy(static_inputs).float().unsqueeze(0)  # (1, pvar_len, H, W)
        target = torch.from_numpy(target).float()                         # (seq_len, out_len, H, W)

        return prev_sm, forcings, static_inputs, target

    def getDataTs(self, ds, varLst, time_range):
        """获取时间序列数据，返回 [nfile, nt, nvar, lat, lon]"""
        if len(ds.dims) == 3:  # 强迫：只有时间维
            nfile = 1
        else:
            nfile = self.nfile

        nvar = len(varLst)
        # 计算时间范围内的天数
        nt_total = len(pd.date_range(time_range[0], time_range[1]))

        data = np.ndarray([nfile, nt_total, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):
            var = ds[varLst[k]]
            if len(ds.dims) > 4:  # model output [file, time, layer, lat, lon]
                dataTemp = var.loc[self.fileidx[0]:self.fileidx[1], time_range[0]:time_range[1], 0, :, :].values
            else:
                dataTemp = var.loc[time_range[0]:time_range[1], :, :].values

            if self.is_train:
                mean = np.nanmean(dataTemp)
                std = np.nanstd(dataTemp)
                self.stat[varLst[k] + '_mean'] = mean
                self.stat[varLst[k] + '_std'] = std
            else:
                mean = self.stat[varLst[k] + '_mean']
                std = self.stat[varLst[k] + '_std']

            dataTemp = (dataTemp - mean) / std

            if len(ds.dims) == 4:  # forcing
                # 对于 forcing，所有文件共用第0个文件的数据
                data[0, :, k, :, :] = dataTemp
            else:
                data[:, :, k, :, :] = dataTemp

        return data

    def getDataConst(self, ds, varLst, time_range):
        """获取静态参数数据，返回 [nfile, nt, nvar, lat, lon]（时间维重复）"""
        nvar = len(varLst)
        nfile = self.nfile
        nt_total = len(pd.date_range(time_range[0], time_range[1]))

        # 先读取静态参数，形状 [nfile, nvar, lat, lon]
        data_static = np.ndarray([nfile, nvar, len(ds.lat), len(ds.lon)])
        for k in range(nvar):
            if varLst[k] == 'expt':
                dataTemp = ds[varLst[k]].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1], 1, :, :]
            elif varLst[k] == 'd2':
                dataTemp = ds['depth'].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1], 1, :, :]
            elif varLst[k] == 'd3':
                dataTemp = ds['depth'].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1], 2, :, :]
            else:
                dataTemp = ds[varLst[k]].values
                dataTemp = dataTemp[self.fileidx[0]:self.fileidx[1], :, :]

            if self.is_train:
                mean = np.nanmean(dataTemp)
                std = np.nanstd(dataTemp)
                self.stat[varLst[k] + '_mean'] = mean
                self.stat[varLst[k] + '_std'] = std
            else:
                mean = self.stat[varLst[k] + '_mean']
                std = self.stat[varLst[k] + '_std']

            dataTemp = (dataTemp - mean) / std
            data_static[:, k, :, :] = dataTemp

        # 在时间维上重复到 nt_total
        out = np.repeat(np.reshape(data_static, [nfile, 1, nvar, len(ds.lat), len(ds.lon)]), nt_total, axis=1)
        return out
# %%
