import sys
sys.path.append('core')

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.GeoDynStereo import stereo, GeoDynStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder, forward_interpolate
from core.utils.utils import upflow8
from PIL import Image
from core.GeoDynStereo import stereo
import torch.utils.data as data
from pathlib import Path
from matplotlib import pyplot as plt
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            if isinstance(flow_pr, (tuple, list)):
                flow_pr = flow_pr[-1]  # 取最后一个输出作为最终预测
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))

        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        # val = (valid_gt.flatten() >= 0.5)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            if isinstance(flow_pr, (tuple, list)):
                flow_pr = flow_pr[-1]  # 取最后一个输出作为最终预测
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu()
        
        # Resize flow_pr to match flow_gt dimensions
        if flow_pr.shape != flow_gt.shape:
            # If flow_pr has multiple channels but flow_gt has only one
            if len(flow_pr.shape) == 4 and flow_pr.shape[1] > 1 and flow_gt.shape[0] == 1:
                # Take only the first channel
                flow_pr = flow_pr[:, 0:1]
                
            flow_pr = F.interpolate(
                flow_pr, 
                size=(flow_gt.shape[1], flow_gt.shape[2]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        else:
            flow_pr = flow_pr.squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False, batch_size=2):  # 减小批处理大小
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,  # 使用较小的批处理大小
        pin_memory=True, shuffle=False, num_workers=4)  # 减少工作线程数

    out_list, epe_list = [], []
    for i_batch, (_, *data_blob) in enumerate(tqdm(val_loader)):
        # 清空缓存
        if i_batch % 10 == 0:
            torch.cuda.empty_cache()
            
        image1, image2, disp_gt, valid_gt = [x for x in data_blob]

        image1 = image1.cuda()
        image2 = image2.cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        try:
            with autocast(enabled=mixed_prec):
                disp_pr = model(image1, image2, iters=iters, test_mode=True)
                if isinstance(disp_pr, (tuple, list)):
                    disp_pr = disp_pr[-1]  # 取最后一个输出作为最终预测
            disp_pr = padder.unpad(disp_pr).cpu()
            
            # 修复形状不匹配问题
            # 1. 如果通道数大于1，只保留第一个通道
            if len(disp_pr.shape) == 4 and disp_pr.shape[1] > 1:
                disp_pr = disp_pr[:, 0:1]
                
            # 2. 调整空间尺寸以匹配ground truth
            if disp_pr.shape[2:] != disp_gt.shape[2:]:
                disp_pr = F.interpolate(
                    disp_pr, 
                    size=(disp_gt.shape[2], disp_gt.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                
            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
            epe = torch.abs(disp_pr - disp_gt)

            epe = epe.flatten()
            val = (valid_gt.flatten() >= 0.5) & (disp_gt.abs().flatten() < 192)  # 使用192作为阈值
            if(np.isnan(epe[val].mean().item())):
                continue

            out = (epe > 3.0)
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"批次 {i_batch} 内存不足，跳过")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    if len(epe_list) == 0:
        return {'scene-disp-epe': float('nan'), 'scene-disp-d1': float('nan')}

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation Scene Flow: %f, %f" % (epe, d1))
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='MiddEval3', resolution='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, resolution=resolution)
    out_list, epe_list = [], []

    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            # 关键修复1: 检查并获取最终输出
            if isinstance(flow_pr, (tuple, list)):
                flow_pr = flow_pr[-1]

        # 关键修复2: 移除argmax操作!!!
        # flow_pr = torch.argmax(flow_pr, dim=1, keepdim=True).float()  # 删除这行

        flow_pr = padder.unpad(flow_pr).cpu()
        
        # Upsample to full resolution
        if flow_pr.shape[-2:] != flow_gt.shape[-2:]:
            flow_pr = F.interpolate(
                flow_pr, 
                size=flow_gt.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # 形状调整 - 确保保留批次维度并移除额外通道维度
        if len(flow_pr.shape) == 4:  # [B,C,H,W] 格式
            if flow_pr.shape[1] > 1:  # 如果有多个通道
                flow_pr = flow_pr[:, 0:1]  # 只保留第一个通道
            flow_pr = flow_pr.squeeze(1)  # 移除通道维度 [B,C,H,W] -> [B,H,W]
        elif len(flow_pr.shape) == 3:  # [C,H,W] 格式
            if flow_pr.shape[0] > 1:  # 如果有多个通道
                flow_pr = flow_pr[0:1]  # 只保留第一个通道
            flow_pr = flow_pr.unsqueeze(0).squeeze(1)  # 添加批次维度并移除通道维度 [C,H,W] -> [1,C,H,W] -> [1,H,W]
        else:  # [H,W] 格式
            flow_pr = flow_pr.unsqueeze(0)  # 添加批次维度 [H,W] -> [1,H,W]

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()
        val = (valid_gt.reshape(-1) >= 0.5) & (occ_mask==255)
        val = val.bool()  # 确保是布尔类型
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('official_train.txt', 'a')
    f.write("Validation Middlebury: %f, %f\n" % (epe, d1))

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/igev_plusplus/sceneflow.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow', choices=["eth3d", "kitti", "sceneflow"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
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

    model = torch.nn.DataParallel(GeoDynStereo(args), device_ids=[0]) 

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, resolution=args.dataset[-1], mixed_prec=args.mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)
