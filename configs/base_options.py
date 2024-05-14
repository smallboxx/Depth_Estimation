# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--exp_name',   type=str, default='imia')
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path',    type=str, default='Data')
        parser.add_argument('--dataset',      type=str, default='mydata',
                            choices=['nyudepthv2', 'kitti', 'imagepath', 'mydata'])
        parser.add_argument('--batch_size',   type=int, default=1)
        parser.add_argument('--workers',      type=int, default=8)
        
        # depth configs
        parser.add_argument('--depth_list',     type=float, default=[20.0, 20.0])
        parser.add_argument('--max_depth_eval', type=float, default=20.0)
        parser.add_argument('--min_depth_eval', type=float, default=0.1)        
        parser.add_argument('--do_kb_crop',     type=int, default=1)
        parser.add_argument('--kitti_crop', type=str, default=None,
                            choices=['garg_crop', 'eigen_crop'])

        parser.add_argument('--backbone',   type=str, default='swin_base_v2')
        parser.add_argument('--pretrained',    type=str, default='weights/swin_v2_base_simmim.pth')
        
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
        parser.add_argument('--seed',  type=int, default=1)  

        parser.add_argument('--finetune_model',    type=str, default=None)
        parser.add_argument('--finetune',    type=bool, default=False)
        return parser
