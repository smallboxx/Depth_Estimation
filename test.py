# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Shift window testing and flip testing is modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# -----------------------------------------------------------------------------

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
from predict_dataset import PreDataset


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    # experiments setting
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.result_dir, args.exp_name)
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

    print("\n2. Define Dataloader")
    if args.dataset == 'imagepath': # not for do_evaluate in case of imagepath
        dataset_kwargs = {'dataset_name': 'ImagePath', 'data_path': args.data_path}
    else:
        dataset_kwargs = {'data_path': args.data_path, 'dataset_name': args.dataset,
                          'is_train': False,'is_eval':args.do_evaluate}

    test_dataset = get_dataset(**dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             pin_memory=True)
    # test_dataset = PreDataset(args)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
    #                          pin_memory=True)

    print("\n3. Inference & Evaluate")
    num =0
    N = len(test_loader)
    for batch_idx, batch in enumerate(test_loader):
        num+=1
        print(f'{num}/{N}')
        input_RGB  = batch['image'].to(device)
        depth_gt   = batch['depth'].to(device)
        filename   = batch['filename'][0]
        domain_gt  = batch['domain'].to(device)
        with torch.no_grad():
            pred = model(input_RGB)
        pred_d = pred['pred_d']

        if args.do_evaluate:
            depth_gt = batch['depth'].to(device)
            pred_d, depth_gt = pred_d.squeeze(), depth_gt.squeeze()
            print(pred_d.shape, depth_gt.shape)
            pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
            computed_result = metrics.eval_depth(pred_crop, gt_crop)
            print(computed_result)
            for metric in metric_name:
                result_metrics[metric] += computed_result[metric]

        if args.save_eval_pngs:
            save_path = os.path.join(result_path, filename)
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            if args.dataset == 'nyudepthv2':
                pred_d = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                pred_d = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
        if args.save_visualize:
            save_path = os.path.join(result_path, filename)
            pred_d_numpy = pred_d.squeeze().cpu().numpy()
            pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            pred_d_numpy = pred_d_numpy.astype(np.uint8)
            pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_RAINBOW)
            print(save_path)
            cv2.imwrite(save_path, pred_d_color)

    if args.do_evaluate:
        for key in result_metrics.keys():
            result_metrics[key] = result_metrics[key] / (batch_idx + 1)
        display_result = logging.display_result(result_metrics)
        if args.kitti_crop:
            print("\nCrop Method: ", args.kitti_crop)
        print(display_result)

    print("Done")


if __name__ == "__main__":
    main()
