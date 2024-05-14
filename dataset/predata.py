import os
import cv2

from dataset.base_dataset import BaseDataset
import numpy as np
import torch.nn.functional as F
import torch


class predata(BaseDataset):
    def __init__(self, data_path, filenames_path='Data',
                 is_train=True, is_eval=True, crop_size=(480, 480), scale_size=True):
        super().__init__(crop_size)
        self.scale_size  = scale_size
        self.is_train    = is_train
        self.is_eval     = is_eval
        self.image_path_list = []
        self.depth_path_list = []

        if is_eval:
            fpathlist = sorted(os.listdir('depth_test/data_sample/test/'))
            self.filenames_list  = ['depth_test/data_sample/test/'+x for x in fpathlist]
            dfpathlist = sorted(os.listdir('depth_test/data_sample/ground_truth/'))
            self.dfpathlist = ['depth_test/data_sample/ground_truth/'+x for x in dfpathlist]
        else:
            fpathlist = sorted(os.listdir('depth_test/preliminary_a/'))
            self.filenames_list  = ['depth_test/preliminary_a/'+x for x in fpathlist]

        phase = 'train' if is_train else 'test'
        print("Dataset: Mydata")
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        if self.is_eval:
            img_path = self.filenames_list[idx]
            gt_path  = self.dfpathlist[idx]

            filename = img_path.split('/')[-2] + '_' + img_path.split('/')[-1]
            image = cv2.imread(img_path)
            assert image is not None, f"Failed to read image! path:{img_path}"
            image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')
            assert depth is not None, f"Failed to read depth! path:{gt_path}"

            if self.is_train:
                image, depth = self.augment_training_data(image, depth)
            else:
                image, depth = self.augment_test_data(image, depth)

            if self.is_train:
                depth[depth > 23000] = 0
                depth = depth / 512.0
            else:
                depth = depth / 1000.0

            max_dp = depth.max() / 15.0
            domain = torch.tensor([max_dp], dtype=torch.float32)
            
            return {'image': image, 'depth': depth, 'domain':domain, 'filename': filename}
        

