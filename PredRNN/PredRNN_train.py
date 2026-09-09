import sys
import os

project_dir = '/home/shengsy/sun2023/zrun/'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PredRNN_args import FSTRParser
from PredRNN_dataset import FSTRDataset
from PredRNN import ForcedSTRNN
import pandas as pd
import matplotlib.pyplot as plt


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        pred_dx = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        target_dx = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)
        return loss_dx + loss_dy


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        return F.huber_loss(pred, target, delta=self.delta)


def train_with_early_stopping(model, train_loader, valid_loader, cfg, fvar_len, pvar_len,
                               epochs, patience=15, grad_weight=0.02, decouple_weight=0.1,
                               loss_type='mse', stage_name="training"):
    from torch.utils.tensorboard import SummaryWriter
    from torch.cuda.amp import autocast, GradScaler

    writer = SummaryWriter(cfg.tensorb_dir)
    valid_losses = []
    scaler = GradScaler()

    if loss_type == 'huber':
        criterion = HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()

    criterion_grad = GradientLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.8, 0.95),
        weight_decay=cfg.weight_decay
    )

    # CosineAnnealingWarmRestarts: 更平滑的学习率调度
    # T_0=10: 第一个周期10个epoch, T_mult=2: 每个后续周期翻倍
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=cfg.lr * 0.01
    )

    best_valid_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(1, epochs + 1):
        if early_stop:
            print(f"\n=== 早停触发！连续 {patience} 个epoch验证损失没有下降 ===")
            break

        model.train()
        epoch_train_losses = []
        epoch_mse_losses = []
        epoch_decouple_losses = []
        epoch_grad_losses = []

        print(f"[{stage_name}] Epoch {epoch}/{epochs}  lr={scheduler.get_last_lr()[0]:.2e}")
        for batch_idx, (vicin, vicout, vicprev) in enumerate(train_loader):
            forcings = vicin[:, :, :fvar_len, :, :].to(cfg.device)
            static_inputs = vicin[:, :, fvar_len:, :, :].to(cfg.device)
            static_inputs = static_inputs[:, 0:1, :, :, :]
            init_cond = vicprev[:, 0:1, :, :, :].to(cfg.device)
            target = vicout.to(cfg.device)

            optimizer.zero_grad()
            with autocast():
                output = model(forcings, init_cond, static_inputs)
                loss_main = criterion(output, target)
                loss_grad = criterion_grad(output, target)
                loss_decouple = model.decouple_loss if hasattr(model, 'decouple_loss') else torch.tensor(0.0, device=cfg.device)
                total_loss = loss_main + decouple_weight * loss_decouple + grad_weight * loss_grad

            scaler.scale(total_loss).backward()
            # 梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            epoch_train_losses.append(total_loss.item())
            epoch_mse_losses.append(loss_main.item())
            epoch_decouple_losses.append(loss_decouple.item())
            epoch_grad_losses.append(loss_grad.item())

            if batch_idx % 200 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, "
                      f"total={total_loss.item():.4f}, mse={loss_main.item():.4f}, "
                      f"decouple={loss_decouple.item():.4f}, grad={loss_grad.item():.4f}")

        avg_train_loss = np.mean(epoch_train_losses)
        avg_mse = np.mean(epoch_mse_losses)
        avg_decouple = np.mean(epoch_decouple_losses)
        avg_grad = np.mean(epoch_grad_losses)

        # 验证
        model.eval()
        test_loss = 0
        test_decouple = 0
        num_batches = len(valid_loader)
        with torch.no_grad():
            for vicin, vicout, vicprev in valid_loader:
                forcings = vicin[:, :, :fvar_len, :, :].to(cfg.device)
                static_inputs = vicin[:, :, fvar_len:, :, :].to(cfg.device)
                static_inputs = static_inputs[:, 0:1, :, :, :]
                init_cond = vicprev[:, 0:1, :, :, :].to(cfg.device)
                target = vicout.to(cfg.device)
                with autocast():
                    output = model(forcings, init_cond, static_inputs)
                    loss = criterion(output, target)
                    dloss = model.decouple_loss if hasattr(model, 'decouple_loss') else torch.tensor(0.0, device=cfg.device)
                test_loss += loss.item()
                test_decouple += dloss.item()
        test_loss /= num_batches
        test_decouple /= num_batches

        print(f"{stage_name} epoch: {epoch:3d}, "
              f"train_loss: {avg_train_loss:.6f} (mse={avg_mse:.4f}, dec={avg_decouple:.4f}, grad={avg_grad:.4f}), "
              f"valid_mse: {test_loss:.6f}, valid_dec: {test_decouple:.4f}")
        writer.add_scalar('loss/train_total', avg_train_loss, epoch)
        writer.add_scalar('loss/train_mse', avg_mse, epoch)
        writer.add_scalar('loss/train_decouple', avg_decouple, epoch)
        writer.add_scalar('loss/valid_mse', test_loss, epoch)
        writer.add_scalar('loss/valid_decouple', test_decouple, epoch)
        valid_losses.append(test_loss)

        if test_loss < best_valid_loss:
            best_valid_loss = test_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  >>> 验证损失下降: {best_valid_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"  --- 验证损失未下降 ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                early_stop = True

        # 定期保存
        if epoch % cfg.epoch_save == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.out_model_dir, f"model_epoch{epoch}.pt"))

    if best_model_state is not None:
        print(f"\n恢复最佳模型 (验证损失: {best_valid_loss:.6f})")
        model.load_state_dict(best_model_state)

    writer.close()
    return valid_losses, best_valid_loss


if __name__ == '__main__':
    cfg = FSTRParser().parse()
    base_model_dir = '/home/shengsy/sun2023/fstr_run_batch400'

    # 训练/验证时间范围
    trange_t = ['2009-01', '2010-12']
    trange_v = ['2013-01', '2014-12']
    fileidx_t = [0, 400]
    fileidx_v = [1200, 1400]

    fvarLst = ['prcp', 'tas', 'dswrf', 'dlwrf', 'pres', 'wind', 'vp']
    pvarLst = ['infilt', 'Ds', 'Dsmax', 'Ws', 'expt', 'd2', 'd3']
    ovarLst = ['OUT_SOIL_MOIST']
    prevLst = ['OUT_SOIL_MOIST']

    # 模型配置
    num_layers = 2
    num_hidden = [48, 48]
    seq_len = 14

    img_channel = len(ovarLst)
    act_channel = len(fvarLst)
    init_cond_channel = img_channel
    static_channel = len(pvarLst)
    out_channel = len(ovarLst)
    filter_size = 5
    stride = 1

    print(f"\n{'='*60}")
    print(f"FSTR 训练 (修复版) | num_hidden={num_hidden} | seq_len={seq_len}")
    print(f"{'='*60}")

    stage_out_dir = os.path.join(base_model_dir, "model_h48_400")
    os.makedirs(stage_out_dir, exist_ok=True)
    cfg.out_model_dir = stage_out_dir
    cfg.tensorb_dir = os.path.join(base_model_dir, "tb_h48_400")

    cfg.frames_input = seq_len
    cfg.lr = 0.0001          # 降低学习率: 3e-4 -> 1e-4
    cfg.weight_decay = 1e-5  # 添加 L2 正则化防止过拟合
    cfg.epochs = 100
    cfg.patience = 15
    cfg.epoch_save = 1

    print(f"学习率: {cfg.lr}, weight_decay: {cfg.weight_decay}")
    print(f"Epochs: {cfg.epochs}, Patience: {cfg.patience}")

    print("\n加载数据集...")
    trainset = FSTRDataset(cfg, fvarLst, pvarLst, ovarLst, prevLst, fileidx_t, trange_t, is_train=True)
    validset = FSTRDataset(cfg, fvarLst, pvarLst, ovarLst, prevLst, fileidx_v, trange_v, is_train=False)

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"训练集: {len(trainset)} 样本, 验证集: {len(validset)} 样本")

    model = ForcedSTRNN(
        num_layers=num_layers,
        num_hidden=num_hidden,
        img_channel=img_channel,
        act_channel=act_channel,
        init_cond_channel=init_cond_channel,
        static_channel=static_channel,
        out_channel=out_channel,
        filter_size=filter_size,
        stride=stride
    )
    model.to(cfg.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    valid_losses, best_loss = train_with_early_stopping(
        model, train_loader, valid_loader, cfg,
        fvar_len=len(fvarLst), pvar_len=len(pvarLst),
        epochs=cfg.epochs, patience=cfg.patience,
        grad_weight=0.02, decouple_weight=0.1,
        loss_type='mse',
        stage_name="H48_400"
    )

    final_model_path = os.path.join(stage_out_dir, "model_epoch_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n训练完成，模型保存至: {final_model_path}")

    # 保存验证损失历史
    df = pd.DataFrame({'epoch': list(range(1, len(valid_losses)+1)), 'valid_loss': valid_losses})
    df.to_csv(os.path.join(stage_out_dir, 'validation_losses.csv'), index=False)

    # ========== 测试 ==========
    print("\n" + "="*50)
    print("开始测试...")
    print("="*50)

    cfg.frames_input = seq_len
    testset = FSTRDataset(cfg, fvarLst, pvarLst, ovarLst, prevLst,
                          [1400, 1500], ['2015-01', '2015-12-26'], is_train=False)

    model.eval()
    nfile = testset.nfile
    nt_use = min(120, testset.nt)
    H, W = testset.forcings.shape[3], testset.forcings.shape[4]

    preds_all = np.zeros((nfile, nt_use, H, W), dtype=np.float32)
    targets_all = np.zeros((nfile, nt_use, H, W), dtype=np.float32)

    with torch.no_grad():
        for file_idx in range(nfile):
            if file_idx % 20 == 0:
                print(f"  测试进度: {file_idx+1}/{nfile}")

            forcings_file = testset.forcings[file_idx][:nt_use]
            static_file = testset.pars[file_idx]
            target_file = testset.target[file_idx][:nt_use]
            prev_file = testset.prev[file_idx]

            init_cond = prev_file[0]

            static_tensor = torch.from_numpy(static_file[0]).float().unsqueeze(0).unsqueeze(0).to(cfg.device)
            forcings_tensor = torch.from_numpy(forcings_file).float().unsqueeze(0).to(cfg.device)
            init_cond_tensor = torch.from_numpy(init_cond).float().unsqueeze(0).unsqueeze(0).to(cfg.device)

            output = model(forcings_tensor, init_cond_tensor, static_tensor)
            preds_all[file_idx] = output[0, :, 0].cpu().numpy()
            targets_all[file_idx] = target_file[:, 0]

    mean = testset.stat['OUT_SOIL_MOIST_mean']
    std = testset.stat['OUT_SOIL_MOIST_std']
    preds_all = preds_all * std + mean
    targets_all = targets_all * std + mean

    epsilon = 1e-8
    kge_per_time = []

    for t in range(nt_use):
        pred_t = preds_all[:, t, :, :].flatten()
        target_t = targets_all[:, t, :, :].flatten()

        r = np.corrcoef(pred_t, target_t)[0, 1]
        alpha = pred_t.mean() / (target_t.mean() + epsilon)
        beta = pred_t.std() / (target_t.std() + epsilon)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        kge_per_time.append(kge)

    kge_mean = np.mean(kge_per_time)

    print(f"\n{'='*50}")
    print(f"========== h48 模型测试结果 ==========")
    print(f"{'='*50}")
    print(f"平均 KGE (120天): {kge_mean:.4f}")
    print(f"5天 KGE: {np.mean(kge_per_time[:5]):.4f}")
    print(f"10天 KGE: {np.mean(kge_per_time[:10]):.4f}")
    print(f"30天 KGE: {np.mean(kge_per_time[:30]):.4f}")
    print(f"60天 KGE: {np.mean(kge_per_time[:60]):.4f}")

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, nt_use+1), kge_per_time, 'g-o', label=f'h48_fixed, KGE={kge_mean:.4f}')
    plt.xlabel('Day')
    plt.ylabel('KGE')
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()
    plt.title(f'KGE (num_hidden={num_hidden}, fixed)')
    plt.savefig(os.path.join(stage_out_dir, 'KGE_h48_fixed.jpg'), dpi=300)
    plt.close()

    print(f"\n最佳验证损失: {best_loss:.6f}")
# %%
