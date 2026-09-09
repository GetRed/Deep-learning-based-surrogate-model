#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import os

from step6_EarlyStopping import EarlyStopping
#%%
def trainmodel( model: nn.Module,  #待训练的 LSTM 模型
                optimizer: torch.optim.Optimizer, #优化器更新模型参数
                loss_func: nn.Module, #损失函数
                train_loader: DataLoader, #加载训练数据
                valid_loader: DataLoader, #验证数据
                cfg):

    writer = SummaryWriter(cfg.tensorb_dir) # 初始化 TensorBoard 日志写入器
    size = len(train_loader.dataset)   # 训练集总样本数
    # early_stopping = EarlyStopping(cfg.patience, verbose=False, delta=0.0001) # 早停实例：耐心值 cfg.patience（7轮），误差下降小于 0.0001 视为无提升

    # 新增：用于保存验证损失的列表
    valid_losses = []

    for epoch in range(1, cfg.epochs+1): # 从 1 到 cfg.epochs（200轮）
        model.train()  # 模型设为训练模式
        

        # early_stopping = EarlyStopping(cfg.patience, verbose=False, delta=0.0001)
        for step, (vicin, vicout) in enumerate(train_loader):   # 遍历训练集批次 # vicin: (batch, time_step, num of variables)  vicout: (batch, time_step)    
            # model.train()

            vicin, vicout = vicin.to(cfg.device), vicout.to(cfg.device) # 数据移到 GPU/CPU

            optimizer.zero_grad()   # 清空上一轮梯度（避免梯度累积）   # clear gradients for this training step
            output = model(vicin)    # 模型前向传播            # lstm output
            loss = loss_func(output, vicout)       # 计算损失         
            loss.backward()          # 反向传播：计算梯度     # backpropagation, compute gradients
            optimizer.step()       # 优化器更新模型参数
            
        ##########################################################
            # writer.add_scalar('loss/train', loss, step)
            # current = step * len(vicin)           

            # # early stop, 每n个batch计算validation dataset的评价
            # if (step+1) % cfg.nbatch_valid == 0:
            #     loss_valid = validmodel(model, loss_func, valid_loader, cfg)
            #     early_stopping(loss_valid, model, cfg)

            #     print(f"epoch: {epoch}, trainloss: {loss:>7f}, [{current:>5d}/{size:>5d}]")
            #     print(f"Avgvalidloss: {loss_valid:>8f} \n")
            #     writer.add_scalar('loss/valid', loss_valid, step) 
                
            #     if early_stopping.early_stop: # 若满足 early stopping 条件
            #         print("Early stopping")
            #         break
        ##########################################################

        if epoch % cfg.epoch_save == 0: # 每 cfg.epoch_save（5轮）保存一次模型
            torch.save(model.state_dict(), cfg.out_model_dir + "model_epoch{}.pt".format(epoch))  # # 保存模型权重（state_dict() 仅存参数，便于后续加载） save model.state_dict() is robust

        ##########################################################
        writer.add_scalar('loss/train', loss, epoch) # 记录当前 epoch 的训练损失（TensorBoard 可视化）
        
        loss_valid = validmodel(model, loss_func, valid_loader, cfg) # 调用验证函数，计算验证集平均损失

        print(f"epoch: {epoch}, trainloss: {loss:>7f}") # 训练损失（最后一个批次的损失）
        print(f"Avgvalidloss: {loss_valid:>8f} \n") # 验证集平均损失
        writer.add_scalar('loss/valid', loss_valid, epoch) # 记录验证损失（TensorBoard）

        # 新增：保存验证损失到列表
        valid_losses.append(loss_valid)


        # if epoch > 10: #早停判断（epoch>10 后启用）# 前 10 轮不早停
        #     early_stopping(loss_valid, model, cfg)  # 传入验证损失，判断是否早停
        #     if early_stopping.early_stop: # 若满足 early stopping 条件
        #         print("Early stopping")
        #         break
    # 新增：训练结束后保存验证损失到CSV
    if valid_losses:  # 确保列表不为空
        # 1. 保存验证损失到CSV文件
        epochs = list(range(1, len(valid_losses) + 1))
        df = pd.DataFrame({
            'epoch': epochs,
            'avgvalidloss': valid_losses
        })
        # 确保目录存在
        os.makedirs(cfg.out_model_dir, exist_ok=True)
        
        # 保存CSV文件
        csv_path = os.path.join(cfg.out_model_dir, 'validation_losses.csv')
        df.to_csv(csv_path, index=False)
        print(f"验证损失已保存到: {csv_path}")


def validmodel(model: nn.Module, loss_func: nn.Module, loader: DataLoader, cfg):
            
    model.eval() # 模型设为评估模式（关闭 Dropout、BatchNorm 冻结统计量）
    test_loss = 0 # 累计验证损失
    num_batches = len(loader) # 验证集批次数

    with torch.no_grad(): # 禁用梯度计算（验证阶段无需更新参数）
        for (vicin, vicout) in loader: # 遍历验证集批次
            vicin, vicout = vicin.to(cfg.device), vicout.to(cfg.device)  # 数据移到对应设备
            output = model(vicin)  # 前向传播预测
            loss = loss_func(output, vicout) # 计算单批次损失
            test_loss += loss.item() # 累计损失（item() 取出张量值，避免梯度占用）

    test_loss /= num_batches # 计算验证集平均损失（总损失/批次数）
            
    return test_loss # 返回平均损失