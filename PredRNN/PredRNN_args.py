#%%
import os
import argparse
import torch
#%%
class FSTRParser(argparse.ArgumentParser):
    def __init__(self):
        super(FSTRParser, self).__init__(description='FSTR Network v2 (Based on HydroFrame)', allow_abbrev=False)

        self.add_argument('--rundir', type=str, default='/home/shengsy/sun2023/', help='root dir to train')
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0003, help='learning rate (原版是3e-4)')
        self.add_argument('--batch-size', type=int, default=400, help='batch size')
        self.add_argument('--batch-size-valid', type=int, default=400, help='validation batch size')
        self.add_argument('--frames_input', type=int, default=10, help='sum of input frames')
        self.add_argument('--frames_output', type=int, default=3, help='sum of predict frames')
        self.add_argument('--patience', type=int, default=7, help='how long to wait after last time validation loss improved')
        self.add_argument('--epoch-save', type=int, default=5, help='number of epochs to save model')

        # 新增：原版 HydroFrame 的参数
        self.add_argument('--weight-decay', type=float, default=0.0, help='weight decay for AdamW')
        self.add_argument('--betas', type=tuple, default=(0.8, 0.95), help='AdamW betas')

    def parse(self):
        # cfg = self.parse_args()
        cfg = self.parse_known_args()[0]

        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg.nbatch_valid = 400

        run_name = f'fstr_run_batch{cfg.batch_size}/'
        cfg.input_path = cfg.rundir + 'input_new/'
        cfg.out_dir = '/home/shengsy/sun2023/surro_dL/HPCresults/FSTR/AE_7pnewin/'
        cfg.out_model_dir = cfg.rundir + run_name + 'model/'
        cfg.tensorb_dir = cfg.rundir + run_name + 'tb/'

        if os.path.exists(cfg.out_dir) is False:
            os.makedirs(cfg.out_dir, exist_ok=True)
            os.makedirs(cfg.out_model_dir, exist_ok=True)
            os.makedirs(cfg.tensorb_dir, exist_ok=True)

        return cfg
#%%