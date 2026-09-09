#%% FSTR 模型测试函数 - 自回归预测
import torch
import numpy as np
from PredRNN_args import FSTRParser
from PredRNN_dataset import FSTRDataset
from PredRNN import ForcedSTRNN


def FSTRtest(cfg, dataset, model_path, num_hidden=[48, 48], num_layers=2, seq_len=14):
    fvar_len = 7
    pvar_len = 7

    model = ForcedSTRNN(
        num_layers=num_layers, num_hidden=num_hidden,
        img_channel=1, act_channel=7,
        init_cond_channel=1, static_channel=7,
        out_channel=1, filter_size=5, stride=1
    ).to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()

    nfile = dataset.nfile
    nt_use = dataset.nt
    H = len(dataset.df.lat)
    W = len(dataset.df.lon)

    preds_all = np.zeros((nfile, nt_use, H, W), dtype=np.float32)
    targets_all = np.zeros((nfile, nt_use, H, W), dtype=np.float32)

    with torch.no_grad():
        for file_idx in range(nfile):
            forcings_file = dataset.forcings[file_idx][:nt_use]
            static_file = dataset.pars[file_idx]
            target_file = dataset.target[file_idx][:nt_use]
            prev_file = dataset.prev[file_idx]

            init_cond = prev_file[0].copy()

            static_tensor = torch.as_tensor(static_file[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(cfg.device)
            forcings_tensor = torch.as_tensor(forcings_file, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            init_cond_tensor = torch.as_tensor(init_cond, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(cfg.device)

            output = model(forcings_tensor, init_cond_tensor, static_tensor)
            preds_all[file_idx] = output[0, :, 0].cpu().numpy()
            targets_all[file_idx] = target_file[:, 0]

    # 反归一化
    mean = dataset.stat['OUT_SOIL_MOIST_mean']
    std = dataset.stat['OUT_SOIL_MOIST_std']
    preds_all = preds_all * std + mean
    targets_all = targets_all * std + mean

    return preds_all, targets_all
# %%
