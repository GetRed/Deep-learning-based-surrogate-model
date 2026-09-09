#%%
import os
import argparse
import torch
#%%
class UNetParser(argparse.ArgumentParser):
    def __init__(self):
        super(UNetParser, self).__init__(description='Dense Encoder-Decoder Convolutional Network')

        self.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
        self.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
        self.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
        self.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')

        #self.add_argument('--rundir', type=str, default='/home/sunrc/work/VICcases/huaihe_new/calibdL/', help='root dir to train')
        self.add_argument('--rundir', type=str, default='/home/shengsy/sun2023/', help='root dir to train')
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0001, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=400, help='batch size')
        self.add_argument('--batch-size-valid', type=int, default=400, help='validation batch size')            
        self.add_argument('--patience', type=int, default=7, help='how long to wait after last time validation loss improved')
        self.add_argument('--epoch-save', type=int, default=5, help='number of epochs to save model')

    def parse(self):
        # cfg = self.parse_args()
        cfg = self.parse_known_args()[0]

        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg.nbatch_valid = 400

        run_name = f'surro_run_batch{cfg.batch_size}/'
        cfg.input_path = cfg.rundir + 'input_new/'
        # cfg.input_path = cfg.rundir + 'input/'
        # cfg.out_dir = cfg.rundir + run_name
        cfg.out_dir = '/home/shengsy/sun2023/surro_dL/HPCresults/UNet/AE_7pnewin/'
        # cfg.out_dir = '/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/paper_ARnet/sample_400/'
        cfg.out_model_dir = cfg.rundir + run_name + 'model/'
        cfg.tensorb_dir = cfg.rundir + run_name + 'tb/'

        if os.path.exists(cfg.out_dir) is False:
            os.mkdir(cfg.out_dir)
            os.mkdir(cfg.out_model_dir)
            os.mkdir(cfg.tensorb_dir)

        return cfg
# %%
class ResNetParser(argparse.ArgumentParser):
    def __init__(self):
        super(ResNetParser, self).__init__(description='Dense Encoder-Decoder Convolutional Network')

        self.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
        self.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
        self.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
        self.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')

        #self.add_argument('--rundir', type=str, default='/home/sunrc/work/VICcases/huaihe_new/calibdL/', help='root dir to train')
        self.add_argument('--rundir', type=str, default='/home/shengsy/sun2023/', help='root dir to train')
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0001, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=400, help='batch size')
        self.add_argument('--batch-size-valid', type=int, default=400, help='validation batch size')            
        self.add_argument('--patience', type=int, default=7, help='how long to wait after last time validation loss improved')
        self.add_argument('--epoch-save', type=int, default=5, help='number of epochs to save model')

    def parse(self):
        # cfg = self.parse_args()
        cfg = self.parse_known_args()[0]

        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg.nbatch_valid = 400

        run_name = f'resnet_run_batch{cfg.batch_size}/'
        cfg.input_path = cfg.rundir + 'input_new/'
        # cfg.input_path = cfg.rundir + 'input/'
        # cfg.out_dir = cfg.rundir + run_name
        cfg.out_dir = '/home/shengsy/sun2023/surro_dL/HPCresults/ResNet/AE_7pnewin/'
        # cfg.out_dir = '/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/paper_ARnet/sample_400/'
        cfg.out_model_dir = cfg.rundir + run_name + 'model/'
        cfg.tensorb_dir = cfg.rundir + run_name + 'tb/'

        if os.path.exists(cfg.out_dir) is False:
            os.mkdir(cfg.out_dir)
            os.mkdir(cfg.out_model_dir)
            os.mkdir(cfg.tensorb_dir)

        return cfg
# %%
class ARnetParser(argparse.ArgumentParser):
    def __init__(self):
        super(ARnetParser, self).__init__(description='Dense Encoder-Decoder Convolutional Network')

        self.add_argument('--blocks', type=list, default=(5, 10, 5), help='list of number of layers in each block in decoding net')
        self.add_argument('--growth-rate', type=int, default=40, help='output of each conv')
        self.add_argument('--drop-rate', type=float, default=0, help='dropout rate')
        self.add_argument('--bn-size', type=int, default=8, help='bottleneck size: bn_size * growth_rate')
        self.add_argument('--bottleneck', action='store_true', default=False, help='enable bottleneck in the dense blocks')
        self.add_argument('--init-features', type=int, default=48, help='# initial features after the first conv layer')

        # self.add_argument('--rundir', type=str, default='/home/sunrc/work/VICcases/huaihe_new/calibdL/', help='root dir to train')
        self.add_argument('--rundir', type=str, default='/home/shengsy/sun2023/', help='root dir to train')
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0001, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=400, help='batch size')
        self.add_argument('--batch-size-valid', type=int, default=400, help='validation batch size')            
        self.add_argument('--patience', type=int, default=7, help='how long to wait after last time validation loss improved')
        self.add_argument('--epoch-save', type=int, default=5, help='number of epochs to save model')

    def parse(self):
        # cfg = self.parse_args()
        cfg = self.parse_known_args()[0]

        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cfg.nbatch_valid = 400

        run_name = f'arnet_run_batch{cfg.batch_size}/'
        cfg.input_path = cfg.rundir + 'input_new/'
        # cfg.input_path = cfg.rundir + 'input/'
        # cfg.out_dir = cfg.rundir + run_name
        cfg.out_dir = '/home/shengsy/sun2023/surro_dL/HPCresults/ARnet/AE_7pnewin/'
        # cfg.out_dir = '/home/sunrc/work/VICcases/huaihe_new/surrogatedL/HPCresults/paper_ARnet/sample_400/'
        cfg.out_model_dir = cfg.rundir + run_name + 'model/'
        cfg.tensorb_dir = cfg.rundir + run_name + 'tb/'

        if os.path.exists(cfg.out_dir) is False:
            os.mkdir(cfg.out_dir)
            os.mkdir(cfg.out_model_dir)
            os.mkdir(cfg.tensorb_dir)

        return cfg
#%%
#class FSTRParser(argparse.ArgumentParser):
#    def __init__(self):
#        super(FSTRParser, self).__init__(description='FSTR (Forced Spatio-Temporal RNN) Network')
#        
#        # FSTR模型参数
#        self.add_argument('--num-layers', type=int, default=3, help='number of LSTM layers')
#        self.add_argument('--num-hidden', type=list, default=[64, 64, 64], help='number of hidden channels in each layer')
#        self.add_argument('--filter-size', type=int, default=5, help='filter size for convolutions')
#        self.add_argument('--stride', type=int, default=1, help='stride for convolutions')
#        
#        # 通道数参数（会根据数据集自动设置）
#        self.add_argument('--act-channel', type=int, default=7, help='channel for forcings')
#        self.add_argument('--static-channel', type=int, default=7, help='channel for static parameters')
#        self.add_argument('--init-cond-channel', type=int, default=1, help='channel for initial condition')
#        self.add_argument('--out-channel', type=int, default=1, help='output channel')
#
#        # 训练参数
#        self.add_argument('--rundir', type=str, default='/home/shengsy/sun2023/', help='root dir to train')
#        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
#        self.add_argument('--lr', type=float, default=0.0001, help='learning rate')
#        self.add_argument('--batch-size', type=int, default=400, help='batch size')
#        self.add_argument('--batch-size-valid', type=int, default=400, help='validation batch size')            
#        self.add_argument('--patience', type=int, default=7, help='how long to wait after last time validation loss improved')
#        self.add_argument('--epoch-save', type=int, default=5, help='number of epochs to save model')
#
#    def parse(self):
#        cfg = self.parse_known_args()[0]
#        
#        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#        cfg.nbatch_valid = 400
#
#        run_name = f'fstr_run_batch{cfg.batch_size}/'
#        cfg.input_path = cfg.rundir + 'input_new/'
#        cfg.out_dir = '/home/shengsy/sun2023/surro_dL/HPCresults/FSTR/AE_7pnewin/'
#        cfg.out_model_dir = cfg.rundir + run_name + 'model/'
#        cfg.tensorb_dir = cfg.rundir + run_name + 'tb/'
#
#        if os.path.exists(cfg.out_dir) is False:
#            os.makedirs(cfg.out_dir, exist_ok=True)
#            os.makedirs(cfg.out_model_dir, exist_ok=True)
#            os.makedirs(cfg.tensorb_dir, exist_ok=True)
#
#        return cfg
#%%
class FSTRParser(argparse.ArgumentParser):
    def __init__(self):
        super(FSTRParser, self).__init__(description='FSTR Network', allow_abbrev=False)

        self.add_argument('--rundir', type=str, default='/home/shengsy/sun2023/', help='root dir to train')
        self.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=0.0001, help='learnign rate')
        self.add_argument('--batch-size', type=int, default=400, help='batch size')
        self.add_argument('--batch-size-valid', type=int, default=400, help='validation batch size')
        self.add_argument('--frames_input', type=int, default=10, help='sum of input frames')                 
        self.add_argument('--frames_output', type=int, default=10, help='sum of predict frames')              
        self.add_argument('--patience', type=int, default=7, help='how long to wait after last time validation loss improved')
        self.add_argument('--epoch-save', type=int, default=5, help='number of epochs to save model')

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