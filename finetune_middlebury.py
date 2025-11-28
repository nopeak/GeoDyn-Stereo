import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging

from core.GeoDynStereo import stereo
from core.stereo_datasets import Middlebury
from core.utils.utils import InputPadder

# 添加 count_parameters 函数定义
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    print(f"Valid mask sum: {valid.sum().item()}, total pixels: {valid.numel()}")
    
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < 192)
    
    print(f"After filtering, valid mask sum: {valid.sum().item()}")
    
    # 如果没有有效像素，返回零损失
    if valid.sum() == 0:
        return torch.tensor(0.0, device=flow_gt.device, requires_grad=True)
    
    valid = valid.unsqueeze(1)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        # 确保 flow_preds[i] 是张量
        if isinstance(flow_preds[i], list):
            pred = flow_preds[i][0]
        else:
            pred = flow_preds[i]
            
        # 检查预测是否包含 NaN
        if torch.isnan(pred).any():
            print(f"Warning: NaN detected in prediction {i}")
            # 用零替换 NaN 值
            pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
            
        i_loss = (pred - flow_gt).abs()
        
        # 限制损失值范围，防止极端值
        i_loss = torch.clamp(i_loss, 0.0, 100.0)
        
        # 计算平均损失
        valid_sum = valid.sum() + 1e-8  # 防止除零
        mean_loss = (valid * i_loss).sum() / valid_sum
        print(f"Prediction {i}, mean loss: {mean_loss.item()}")
        
        # 检查损失是否为 NaN
        if torch.isnan(mean_loss):
            print(f"Warning: NaN loss detected for prediction {i}")
            continue  # 跳过这个预测
            
        flow_loss += i_weight * mean_loss

    # 确保最终损失不是 NaN
    if torch.isnan(flow_loss):
        print("Warning: Final loss is NaN, returning zero loss")
        return torch.tensor(0.0, device=flow_gt.device, requires_grad=True)
        
    return flow_loss

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

# 在 finetune 函数中添加梯度累积
def finetune(args):
    model = nn.DataParallel(stereo(args), device_ids=[0])
    logging.info("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)
        logging.info(f"Loaded checkpoint from {args.restore_ckpt}")

    model.cuda()
    model.train()
    
    # 检查模型是否真的处于训练模式
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            if not module.training:
                logging.warning("Some BatchNorm layers are not in training mode!")
    
    # 检查是否有需要梯度的参数
    trainable_params = sum(p.requires_grad for p in model.parameters())
    logging.info(f"Model has {trainable_params} trainable parameters")
    
    # 准备Middlebury数据集
    aug_params = {
        'crop_size': args.image_size, 
        'min_scale': args.spatial_scale[0], 
        'max_scale': args.spatial_scale[1], 
        'do_flip': False
    }
    
    # 使用Middlebury数据集进行微调
    train_dataset = Middlebury(aug_params, split=args.split, resolution=args.resolution)
    logging.info(f"Training with {len(train_dataset)} samples from Middlebury dataset")
    
    # 检查数据集的第一个样本
    sample = train_dataset[0]
    _, img1, img2, disp_gt, valid_gt = sample
    logging.info(f"Sample image shape: {img1.shape}, disparity shape: {disp_gt.shape}")
    logging.info(f"Disparity range: min={disp_gt.min().item()}, max={disp_gt.max().item()}")
    logging.info(f"Valid mask sum: {valid_gt.sum().item()}, total: {valid_gt.numel()}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             pin_memory=True, shuffle=True, num_workers=args.num_workers,
                             drop_last=True)
    
    optimizer, scheduler = fetch_optimizer(args, model)
    
    total_steps = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    logger = SummaryWriter(log_dir=args.log_dir)

    # 添加梯度累积参数
    accumulation_steps = 4  # 累积4个批次的梯度
    
    should_keep_training = True
    while should_keep_training:
        for i_batch, (_, image1, image2, flow_gt, valid_gt) in enumerate(train_loader):
            image1 = image1.cuda()
            image2 = image2.cuda()
            flow_gt = flow_gt.cuda()
            valid_gt = valid_gt.cuda()

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                flow_predictions = model(image1, image2, iters=args.iters)
                loss = sequence_loss(flow_predictions, flow_gt, valid_gt)
                # 除以累积步数
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            # 每 accumulation_steps 步更新一次参数
            if (i_batch + 1) % accumulation_steps == 0 or (i_batch + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()
            
            logger.add_scalar('loss', loss.item(), total_steps)
            logger.add_scalar('learning_rate', scheduler.get_last_lr()[0], total_steps)

            if total_steps % args.print_freq == 0:
                logging.info(f'Step {total_steps}: loss = {loss.item():.6f}')

            if total_steps % args.save_freq == 0:
                PATH = os.path.join(args.output_dir, f'middlebury_finetuned_{total_steps}.pth')
                torch.save(model.state_dict(), PATH)
                logging.info(f"Saved checkpoint to {PATH}")

            total_steps += 1
            if total_steps >= args.num_steps:
                should_keep_training = False
                break

    # 保存最终模型
    PATH = os.path.join(args.output_dir, 'middlebury_finetuned_final.pth')
    torch.save(model.state_dict(), PATH)
    logging.info(f"Saved final checkpoint to {PATH}")
    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_plusplus/sceneflow.pth')
    parser.add_argument('--output_dir', help="directory to save checkpoints", default='./checkpoints')
    parser.add_argument('--log_dir', help="directory to save logs", default='./logs')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0.8, 1.0])
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t use y jitter')
    
    # Middlebury数据集参数
    parser.add_argument('--split', type=str, default='MiddEval3', choices=["2005", "2006", "2014", "2021", "MiddEval3"])
    parser.add_argument('--resolution', type=str, default='F', choices=['F', 'H', 'Q'])
    
    # 模型架构参数
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    # 开始微调
    finetune(args)
