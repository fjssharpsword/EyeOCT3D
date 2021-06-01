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
import scipy.misc as misc
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

def read_dataset():
    img_dir = '/data/fjsdata/OCTA500/OCTA_6M/10001/'
    scanlist=os.listdir(img_dir)
    scanlist=natsort.natsorted(scanlist)
    scan_num=-1
    cube = np.zeros((len(scanlist), 400, 640), dtype=np.uint8)
    for scan in scanlist:
        scan_num += 1
        image = cv2.imread(os.path.join(img_dir, scan), cv2.IMREAD_GRAYSCALE) #gray
        cube[scan_num, :, :] = image.transpose(1,0)
    print(cube.shape)
    mask_dir = '/data/fjsdata/OCTA500/OCTA-500/OCTA_6M/GroundTruth/10001.bmp'
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    #cv2.imwrite('/data/pycode/EyeOCT3D/imgs/mask.jpg', mask)
    mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
    #cv2.imwrite('/data/pycode/EyeOCT3D/imgs/mask_90.jpg', mask)
    print(mask.shape)


if __name__ == "__main__":

    
    read_dataset()