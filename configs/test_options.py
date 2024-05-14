# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

from configs.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        parser = BaseOptions.initialize(self)
        parser.add_argument('--result_dir', type=str, default='./results',
                            help='save result images into result_dir/exp_name')
        parser.add_argument('--ckpt_dir',   type=str,
                            default='logs/0907_mydata_4_swin_v2_large_simmim_deconv3_32_2_480_480_3e-05_5e-06_085_005_50_30_30_30_15_2_2_18_2_frist/best.pth', 
                            help='load ckpt path')
        
        parser.add_argument('--save_eval_pngs', type=bool, default=False,
                            help='save result image into evaluation form')
        parser.add_argument('--save_visualize', type=bool, default=True,
                            help='save result image into visulized form')
        parser.add_argument('--do_evaluate', type=bool, default=True,
                            help='evaluate with inferenced images')   
        parser.add_argument('--crop_h',  type=int, default=480)
        
        return parser


