import torch 
import torch.utils.data as data
import os
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class fisheye_kb4():
    fx = 549.0869836743506
    fy = 549.0393143358732
    cx = 664.8310221457081
    cy = 368.30351397130175
    k1 = -0.037032730499053215
    k2 = -9.331683195791314e-05
    k3 = -0.0025427846701313096
    k4 = 0.0005759176479469663
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]])

    dist_coeffs = np.array([k1, k2, k3, k4])

class PreDataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        if args.do_evaluate:
            fpathlist = os.listdir('final_sample/test/')
            self.fpathlist = fpathlist
        else:
            fpathlist = sorted(os.listdir('final_a/'))
            self.fpathlist  = ['final_a/'+x for x in fpathlist]
        self.do_evaluate = args.do_evaluate
        self.to_tensor = transforms.ToTensor()
        self.fisheye_cam_m= fisheye_kb4.camera_matrix
        self.fisheye_dist = fisheye_kb4.dist_coeffs


    def __len__(self):
        return len(self.fpathlist)
    
    def __getitem__(self, index):
        if self.do_evaluate:
            img_path = 'final_sample/test/'+self.fpathlist[index]
            gt_path  = 'final_sample/ground_truth/'+self.fpathlist[index].replace('jpg', 'png')
            filename = img_path.split('/')[-1]
            image = cv2.imread(img_path)
            assert image is not None, f"Failed to read image! path:{img_path}"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
            image = self.to_tensor(image)
            depth = self.to_tensor(depth)
            depth = depth / 1000.0
            max_dp = depth.max() / 20.0
            domain = torch.tensor([max_dp], dtype=torch.float32)

            return {'image': image, 'depth':depth, 'domain':domain, 'filename': filename}
        
        else:
            img_path = self.fpathlist[index]
            filename = img_path.split('/')[-1]
            image = cv2.imread(img_path)
            assert image is not None, f"Failed to read image! path:{img_path}"
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.to_tensor(image)
            return {'image': image,'filename': filename}
            