import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.GeoDynStereo import GeoDynStereo
from evaluate_stereo import *
import core.stereo_datasets as datasets
import torch.nn.functional as F

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def sequence_loss(args, agg_preds, iter_preds, disp_gt, valid, max_disp=192, max_disp0=192, max_disp1=192, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    # Ensure disp_gt is [B, 1, H, W] and explicitly convert to float
    if disp_gt.ndim == 3:
        disp_gt = disp_gt.unsqueeze(1)
    disp_gt = disp_gt.float()
    
    # Ensure 'valid' is [B, H, W] and explicitly convert to float
    if valid.ndim == 4 and valid.shape[1] == 1: # If valid is [B, 1, H, W]
        valid = valid.squeeze(1) # Make it [B, H, W]
    valid = valid.float()

    # Ensure predictions (agg_preds and iter_preds) are [B, 1, H, W]
    # This assumes agg_preds and iter_preds are lists of tensors.
    agg_preds = [p.unsqueeze(1) if p.ndim == 3 else p for p in agg_preds]
    iter_preds = [p.unsqueeze(1) if p.ndim == 3 else p for p in iter_preds]
    
    # 获取ground truth的批次大小和空间尺寸
    gt_batch, gt_h, gt_w = disp_gt.shape[0], disp_gt.shape[2], disp_gt.shape[3]
    
    # 调整预测结果的批次大小和空间尺寸以匹配ground truth
    agg_preds_resized = []
    for pred in agg_preds:
        # 检查批次大小
        if pred.shape[0] != gt_batch:
            # 如果预测的批次大小为1而ground truth批次大小大于1，则重复预测
            if pred.shape[0] == 1 and gt_batch > 1:
                pred = pred.repeat(gt_batch, 1, 1, 1)
            # 如果预测的批次大小大于ground truth，则截取
            elif pred.shape[0] > gt_batch:
                pred = pred[:gt_batch]
            else:
                # 其他情况，可能需要更复杂的处理
                raise ValueError(f"无法处理批次大小不匹配：pred batch={pred.shape[0]}, gt batch={gt_batch}")
        
        # 检查空间尺寸
        if pred.shape[2:] != (gt_h, gt_w):
            pred_resized = F.interpolate(pred, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
            agg_preds_resized.append(pred_resized)
        else:
            agg_preds_resized.append(pred)
    agg_preds = agg_preds_resized
    
    iter_preds_resized = []
    for pred in iter_preds:
        # 检查批次大小
        if pred.shape[0] != gt_batch:
            # 如果预测的批次大小为1而ground truth批次大小大于1，则重复预测
            if pred.shape[0] == 1 and gt_batch > 1:
                pred = pred.repeat(gt_batch, 1, 1, 1)
            # 如果预测的批次大小大于ground truth，则截取
            elif pred.shape[0] > gt_batch:
                pred = pred[:gt_batch]
            else:
                # 其他情况，可能需要更复杂的处理
                raise ValueError(f"无法处理批次大小不匹配：pred batch={pred.shape[0]}, gt batch={gt_batch}")
        
        # 检查空间尺寸
        # 添加检查确保pred有足够的维度
        if len(pred.shape) < 3:
            print(f"警告: 跳过没有空间维度的预测: {pred.shape}")
            continue
            
        if pred.shape[2:] != (gt_h, gt_w):
            pred_resized = F.interpolate(pred, size=(gt_h, gt_w), mode='bilinear', align_corners=False)
            iter_preds_resized.append(pred_resized)
        else:
            iter_preds_resized.append(pred)
    iter_preds = iter_preds_resized

    n_predictions = len(iter_preds)
    assert n_predictions >= 1
    if ('kitti' in args.train_datasets) or ('eth3d' in args.train_datasets):
        max_disp0 = 192
        max_disp1 = 192
        max_disp = 192
    else:
        max_disp0 = 192
        max_disp1 = 384
        max_disp = 700

    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    mask0 = ((valid >= 0.5) & (mag < max_disp0)).unsqueeze(1)
    mask1 = ((valid >= 0.5) & (mag < max_disp1)).unsqueeze(1)
    mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert mask.shape == disp_gt.shape, [mask.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[mask.bool()]).any()

    # 检查agg_preds的长度并相应处理
    agg_preds_len = len(agg_preds)
    
    # 第一个预测（权重1.0）
    if agg_preds_len > 0:
        disp_loss += 1.0 * F.smooth_l1_loss(agg_preds[0][mask0.bool()], disp_gt[mask0.bool()], reduction='mean')
    
    # 第二个预测（权重0.5）
    if agg_preds_len > 1:
        disp_loss += 0.5 * F.smooth_l1_loss(agg_preds[1][mask1.bool()], disp_gt[mask1.bool()], reduction='mean')
    
    # 第三个预测（权重0.2）
    if agg_preds_len > 2:
        disp_loss += 0.2 * F.smooth_l1_loss(agg_preds[2][mask.bool()], disp_gt[mask.bool()], reduction='mean')
    else:
        # 如果没有第三个预测，使用最后一个可用的预测
        if agg_preds_len > 0:
            last_pred_index = agg_preds_len - 1
            disp_loss += 0.2 * F.smooth_l1_loss(agg_preds[last_pred_index][mask.bool()], disp_gt[mask.bool()], reduction='mean')
            # 删除警告打印语句
            # print(f"警告：agg_preds长度为{agg_preds_len}，使用索引{last_pred_index}代替索引2")
    
    for i in range(n_predictions):
        if n_predictions == 1:
            # If only one prediction, apply full weight or a default strategy
            i_weight = 1.0
        else:
            # n_predictions - 1 will not be zero here
            adjusted_loss_gamma = loss_gamma**(15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        
        i_loss = (iter_preds[i] - disp_gt).abs()
        assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, disp_gt.shape, iter_preds[i].shape]
        disp_loss += i_weight * i_loss[mask.bool()].mean()

    # Calculate EPE for the last prediction
    # iter_preds[-1] and disp_gt are typically [B, 1, H, W]
    # epe_map will be [B, H, W]
    epe_map = torch.sum((iter_preds[-1] - disp_gt)**2, dim=1).sqrt()
    
    # mask is [B, 1, H, W]. Squeeze it to [B, H, W] for applying to epe_map.
    valid_mask_squeezed = mask.squeeze(1).bool()
    epe_values_masked = epe_map[valid_mask_squeezed] # This will be a 1D tensor of valid EPEs

    metrics = {}
    if epe_values_masked.numel() > 0:
        metrics['epe'] = epe_values_masked.mean().item()
        metrics['1px'] = (epe_values_masked < 1).float().mean().item()
        metrics['3px'] = (epe_values_masked < 3).float().mean().item()
        metrics['5px'] = (epe_values_masked < 5).float().mean().item()
    else:
        # Handle case where mask is all false, resulting in empty epe_values_masked
        metrics['epe'] = 0.0
        metrics['1px'] = 0.0
        metrics['3px'] = 0.0
        metrics['5px'] = 0.0

    return disp_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100
    def __init__(self, model, scheduler, logdir):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.logdir = logdir
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.image_log_freq = 500  # 每500步记录一次图像
        self.histogram_log_freq = 1000  # 每1000步记录一次权重直方图
        self.last_image_log = 0
        self.last_histogram_log = 0

    # 添加新方法
    def log_images(self, step, image1, image2, disp_gt, pred_disp, valid):
        """记录图像和视差图到TensorBoard"""
        with torch.no_grad():
            # 只记录批次中的第一个样本
            img1 = image1[0].detach().cpu().numpy().transpose(1, 2, 0)
            img2 = image2[0].detach().cpu().numpy().transpose(1, 2, 0)
            
            # 归一化视差图用于可视化
            disp_gt_norm = (disp_gt[0] - disp_gt[0].min()) / (disp_gt[0].max() - disp_gt[0].min() + 1e-6)
            pred_disp_norm = (pred_disp[0] - pred_disp[0].min()) / (pred_disp[0].max() - pred_disp[0].min() + 1e-6)
            
            # 创建彩色视差图
            disp_gt_color = apply_colormap(disp_gt_norm.squeeze().cpu().numpy())
            pred_disp_color = apply_colormap(pred_disp_norm.squeeze().cpu().numpy())
            
            # 有效掩码可视化
            valid_mask = valid[0].float().cpu().numpy()
            valid_mask_color = np.stack([valid_mask]*3, axis=-1) * 255
            
            # 记录到TensorBoard
            self.writer.add_image('Input/Left_Image', img1, step, dataformats='HWC')
            self.writer.add_image('Input/Right_Image', img2, step, dataformats='HWC')
            self.writer.add_image('Disparity/Ground_Truth', disp_gt_color, step, dataformats='HWC')
            self.writer.add_image('Disparity/Predicted', pred_disp_color, step, dataformats='HWC')
            self.writer.add_image('Mask/Valid_Areas', valid_mask_color, step, dataformats='HWC')
            
            # 创建误差热力图
            error_map = torch.abs(pred_disp[0] - disp_gt[0]).squeeze().cpu().numpy()
            error_map_norm = error_map / (error_map.max() + 1e-6)
            error_heatmap = apply_colormap(error_map_norm)
            self.writer.add_image('Error/Disparity_Error', error_heatmap, step, dataformats='HWC')

    def log_histograms(self, step, model):
        """记录模型权重和梯度的直方图"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    
    def log_learning_rates(self, step, optimizer):
        """记录每个参数组的学习率"""
        for i, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'LearningRate/group_{i}', group['lr'], step)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(GeoDynStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, args.logdir)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")
    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            # 移除固定尺寸调整，直接使用原始数据加载器的尺寸
            # 只需要确保valid维度正确
            if valid.ndim == 3:  # [N,H,W]
                valid = valid.unsqueeze(1)  # [N,1,H,W]
            elif valid.ndim == 4 and valid.shape[1] == 1:  # 已经是[N,1,H,W]
                pass
            else:
                valid = valid.squeeze()  # 移除多余维度
                if valid.ndim == 3:
                    valid = valid.unsqueeze(1)

            agg_preds, iter_preds = model(image1, image2, max_iters=args.train_iters)
            assert model.training

            loss, metrics = sequence_loss(args, agg_preds, iter_preds, disp_gt, valid) # Corrected argument order
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)
                
                # 在验证前清空缓存
                torch.cuda.empty_cache()
                
                # 使用较小的批处理大小进行验证
                if 'sceneflow' in args.train_datasets:
                    try:
                        results = validate_sceneflow(model.module, iters=args.valid_iters)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            logging.warning("CUDA内存不足，跳过验证")
                            results = {"scene-disp-epe": float('nan'), "scene-disp-d1": float('nan')}
                        else:
                            raise e
                elif 'kitti' in args.train_datasets:
                    results = validate_kitti(model.module, iters=args.valid_iters)
                elif 'middlebury' in args.train_datasets:
                    results = validate_middlebury(model.module, iters=args.valid_iters)
                elif 'eth3d' in args.train_datasets:
                    results = validate_eth3d(model.module, iters=args.valid_iters)
                else:
                    print(f"Val dataset is not supported.")
                
                # logger.write_dict(results)
                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--logdir', default='./checkpoints', help='the directory to save logs and checkpoints')
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help="batch size used during training.")
    parser.add_argument('--train_datasets', default='sceneflow', choices=['sceneflow', 'kitti', 'middlebury_train', 'middlebury_finetune', 'eth3d_train', 'eth3d_finetune'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 768], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 128, 128],
                    help="hidden state and context dimensions (MUST provide 3 values)")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48, help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96, help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192, help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1, help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2, help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4, help="disp interval of large disparity-range geometry encoding volume")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4, 0.8], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    train(args)
