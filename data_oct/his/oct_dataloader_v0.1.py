import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
import sys
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import natsort
from scipy.ndimage import zoom
#define by myself
sys.path.append("..") 
from EyeOCT3D.config import *
#from config import *
"""
Dataset:OCTA-500   https://ieee-dataport.org/open-access/octa-500
1) OCTA_6M(No.10001-No.10300):  FOV(field of view)=6mm*6mm*2mm, Volume=400pixel*400pixel*640pixel
2) OCTA_3M(No.10301-No.10500):  FOV=3mm*3mm*2mm, Volume=304pixel*304pixel*640pixel
3) Label: Text label(gender, age, os/od, disease), pixel label(retinal vessel segmentation, foveal avascular zone segmentation)
4) Projection Maps：--OCT FULL(average), --OCT ILM_OPL (average), --OCT OPL_BM (average), --OCTA FULL (average), --OCTA ILM_OPL (maximum), --OCTA OPL_BM (maximum)
5) Disease: age-related macular degeneration (AMD), diabetic retinopathy (DR), choroidal neovascularization (CNV), 
            central serous chorioretinopathy (CSC), retinal vein occlusion (RVO)
Github: https://github.com/chaosallen/IPNV2_pytorch
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_dataset_file):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        vol_ids = []
        vol_labels = []
        vol_imgs = []
        
        scanlist =os.listdir(path_to_dataset_file)
        for scan in scanlist:
            #_, file = os.path.split(scan)
            vol_imgs.append(os.path.join(path_to_dataset_file, scan))
            scan = scan.split('.')[0].split('_')
            vol_ids.append(int(scan[0]))
            vol_labels.append(int(scan[1]))
            
        self.vol_ids = vol_ids
        self.vol_imgs = vol_imgs
        self.vol_labels = vol_labels

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image: Bx2x160x100x100, the two modes are: OCT and OCTA
                   and the mode of 3M and 6M all are resized as 160*100*100
            label:disease, 0=normal, 1=others, 2=DR, 3=AMD, 4=CNV, 5=CSC, 6=RVO
        """
        #get image
        scan =  self.vol_imgs[index]
        cube = torch.load(scan)

        cube = zoom(cube, (1, config['CUBE_SIZE'][0]/cube.shape[1], \
                              config['CUBE_SIZE'][1]/cube.shape[2], \
                              config['CUBE_SIZE'][2]/cube.shape[3]), order=1)
        cube = torch.as_tensor(cube, dtype=torch.float32)

        cube = cube/255 #[0, 255] turn to [0,1]
        #get label
        label = torch.as_tensor(self.vol_labels[index], dtype=torch.long)

        return cube, label

    def __len__(self):
        return len(self.vol_ids)

"""
def collate_fn(batch):
    return tuple(zip(*batch))
 #DataLoader: collate_fn=collate_fn
"""

PATH_TO_TRAIN_FILE = '/data/tmpexec/OCTA500/train/'
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_dataset_file=PATH_TO_TRAIN_FILE)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

PATH_TO_TEST_FILE = '/data/tmpexec/OCTA500/test/'
def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_dataset_file=PATH_TO_TEST_FILE)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":

    #generate train files.   
    datasets = get_train_dataloader(batch_size=10, shuffle=False, num_workers=1)
    for batch_idx, (cube, label) in enumerate(datasets):
        print(cube.shape)
        print(label.shape)
        break