import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import GLPDepth
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import argparse
from predict_dataset import PreDataset


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']



def main(args):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join('./results',args.save_dir)
        logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)
    
    if args.do_evaluate:
        result_metrics = {}
        for metric in metric_name:
            result_metrics[metric] = 0.0

    print("\n1. Define Model")
    model = GLPDepth(args=args).to(device)
    
    model_weight = torch.load(args.ckpt_dir)
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight)
    model.eval()
    model.infer(True)

    print("\n2. Define Dataloader")

    test_dataset = PreDataset(args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\n3. Inference & Evaluate")
    num =0
    N = len(test_loader)
    for batch_idx, batch in enumerate(test_loader):
        
        num+=1
        print(f'{num}/{N}')
        input_RGB = batch['image'].to(device)

        filename = batch['filename']

        with torch.no_grad():
            pred = model(input_RGB)
        pred_d = pred['pred_d']
        max_dp_pred = pred['domain']
        print('pred:',max_dp_pred)
        if args.do_evaluate:
            max_dp_gt= batch['domain'].to(device)
            depth_gt = batch['depth'].to(device)
            print('gt:',max_dp_gt)
            pred_d, depth_gt = pred_d.squeeze(), depth_gt.squeeze()
            pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            print(computed_result)
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]

        if args.save_eval_pngs:
            save_path = os.path.join(result_path, filename[0])
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            pred_d = pred_d.cpu().numpy() * 1000.0
            cv2.imwrite(save_path, pred_d.astype(np.uint16),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            
        if args.save_visualize:
            save_path    = os.path.join(result_path, filename[0])
            gt_save_path = os.path.join(result_path, 'gt_'+filename[0])
            pred_d_numpy = pred_d.squeeze().cpu().numpy()
            pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            pred_d_numpy = pred_d_numpy.astype(np.uint8)
            # pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)

            depth_gt_numpy = depth_gt.squeeze().cpu().numpy()
            depth_gt_numpy = (depth_gt_numpy / depth_gt_numpy.max()) * 255
            depth_gt_numpy = depth_gt_numpy.astype(np.uint8)
            # depth_gt_color = cv2.applyColorMap(depth_gt_numpy, cv2.COLORMAP_RAINBOW)
            cv2.imwrite(save_path, pred_d_numpy)
            cv2.imwrite(gt_save_path, depth_gt_numpy)

    if args.do_evaluate:
        for key in result_metrics.keys():
            result_metrics[key] = result_metrics[key] / (batch_idx + 1)
        display_result = logging.display_result(result_metrics)
        print(display_result)

    print("Done")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def config():
    parser = argparse.ArgumentParser(description='prediction')
    parser.add_argument('--ckpt_dir', type=str, default='best_large_new.pth')
    parser.add_argument('--datapath', type=str, default='depth_test')
    parser.add_argument('--save_dir', type=str, default='predict_weq')
    parser.add_argument('--save_visualize', type=bool, default=True)
    parser.add_argument('--save_eval_pngs', type=bool, default=False)
    parser.add_argument('--do_evaluate', type=bool, default=True)
    parser.add_argument('--max_depth_eval', type=float, default=20.0)
    parser.add_argument('--min_depth_eval', type=float, default=0.1)        
    parser.add_argument('--do_kb_crop',     type=int, default=1)
    parser.add_argument('--kitti_crop', type=str, default=None,
                        choices=['garg_crop', 'eigen_crop'])

    parser.add_argument('--backbone',   type=str, default='swin_large')
    parser.add_argument('--pretrained',    type=str, default=None)
    parser.add_argument('--window_size', type=int, default=[30, 30, 30, 15])
    parser.add_argument('--pretrain_window_size', type=int, default=[12, 12, 12, 6])
    parser.add_argument('--use_shift', type=str2bool, default=[True, True, False, False])
    parser.add_argument('--depths', type=int, default=[2, 2, 18, 2])
    parser.add_argument('--drop_path_rate',     type=float, default=0.5)
    parser.add_argument('--use_checkpoint',   type=str2bool, default='False')
    parser.add_argument('--num_deconv',     type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=[32, 32, 32])
    parser.add_argument('--deconv_kernels', type=int, default=[2, 2, 2])

    parser.add_argument('--shift_window_test', action='store_true')     
    parser.add_argument('--shift_size',  type=int, default=2)
    parser.add_argument('--flip_test', action='store_true')     
    parser.add_argument('--depth_list',      type=float, default=[10.0,14.0])
    parser.add_argument('--crop_h',  type=int, default=480)
    parser.add_argument('--finetune',    type=bool, default=False)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = config()
    main(args)
