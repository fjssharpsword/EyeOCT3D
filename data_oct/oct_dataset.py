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
from config import *
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
    def __init__(self, path_to_dataset_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        vol_ids = []
        vol_imgs = []
        vol_labels = []

        datas = pd.read_csv(path_to_dataset_file, sep=',') # header=None
        for row in datas.values:#dataframe->numpy
            vol_ids.append(row[0]) #ID
            vol_labels.append(row[6]) #Label
            if row[5] == '6M': #type= '6M'
                path_oct = os.path.join(PATH_TO_OCT_6M, str(row[0]))
                path_octa = os.path.join(PATH_TO_OCTA_6M, str(row[0]))
                vol_imgs.append([path_octa, path_oct])
            else: #type= '3M'
                path_oct = os.path.join(PATH_TO_OCT_3M, str(row[0]))
                path_octa = os.path.join(PATH_TO_OCTA_3M, str(row[0]))
                vol_imgs.append([path_octa, path_oct])
        
        self.vol_ids = vol_ids
        self.vol_imgs = vol_imgs
        self.vol_labels = vol_labels
        self.transform = transform

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
        vol_imgs =  self.vol_imgs[index]
        #read octa mode
        scanlist=os.listdir(vol_imgs[0])
        scanlist=natsort.natsorted(scanlist)
        cube_octa = torch.FloatTensor() 
        for scan in scanlist:
            img_dir = os.path.join(vol_imgs[0], scan)
            img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE).transpose(1,0) #gray
            img = zoom(img, (config['CUBE_SIZE'][1]/img.shape[0], config['CUBE_SIZE'][2]/img.shape[1]), order=1)
            img = torch.as_tensor(img, dtype=torch.float32)
            cube_octa = torch.cat((cube_octa, img.unsqueeze(0)), 0)
        assert cube_octa.shape[0] == 304 or cube_octa.shape[0] == 400
        cube_octa = zoom(cube_octa, (config['CUBE_SIZE'][0]/cube_octa.shape[0], 1, 1), order=1)
        cube_octa = torch.as_tensor(cube_octa, dtype=torch.float32)
        #read oct mode
        scanlist=os.listdir(vol_imgs[1])
        scanlist=natsort.natsorted(scanlist)
        cube_oct = torch.FloatTensor() 
        for scan in scanlist:
            img_dir = os.path.join(vol_imgs[1], scan)
            img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE).transpose(1,0) #gray
            img = zoom(img, (config['CUBE_SIZE'][1]/img.shape[0], config['CUBE_SIZE'][2]/img.shape[1]), order=1)
            img = torch.as_tensor(img, dtype=torch.float32)
            cube_oct = torch.cat((cube_oct, img.unsqueeze(0)), 0)
        assert cube_oct.shape[0] == 304 or cube_oct.shape[0] == 400
        cube_oct = zoom(cube_oct, (config['CUBE_SIZE'][0]/cube_oct.shape[0], 1, 1), order=1)
        cube_oct = torch.as_tensor(cube_oct, dtype=torch.float32)
        #merge channel(two mode)
        cube = torch.cat((cube_octa.unsqueeze(0), cube_oct.unsqueeze(0)), 0)  
        #save and load
        label = self.vol_labels[index]
        save_path = '/data/tmpexec/OCTA500/test/' + str(self.vol_ids[index]) + '_' + str(label) +'.pt'
        torch.save(cube, save_path)

        #get label
        label = torch.as_tensor(label, dtype=torch.long)

        return cube, label

    def __len__(self):
        return len(self.vol_ids)

"""
def collate_fn(batch):
    return tuple(zip(*batch))
 #DataLoader: collate_fn=collate_fn
"""

PATH_TO_OCTA_6M = '/data/fjsdata/OCTA500/OCTA_6M/'
PATH_TO_OCT_6M = '/data/fjsdata/OCTA500/OCT_6M/'
PATH_TO_OCTA_3M = '/data/fjsdata/OCTA500/OCTA_3M/'
PATH_TO_OCT_3M = '/data/fjsdata/OCTA500/OCT_3M/'
PATH_TO_TRAIN_FILE = '/data/pycode/EyeOCT3D/data_oct/OCTA500_Train.txt'
def get_train_dataloader(batch_size, shuffle, num_workers):
    dataset_train = DatasetGenerator(path_to_dataset_file=PATH_TO_TRAIN_FILE)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

PATH_TO_TEST_FILE = '/data/pycode/EyeOCT3D/data_oct/OCTA500_Test.txt'
def get_test_dataloader(batch_size, shuffle, num_workers):
    dataset_test = DatasetGenerator(path_to_dataset_file=PATH_TO_TEST_FILE)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

def split_dataset():
    data_path = '/data/pycode/EyeOCT3D/data_oct/OCTA500_label.csv'
    datas = pd.read_csv(data_path, sep=',') 
    #encoding disease as numberic value
    #0=normal, 1=others, 2=DR, 3=AMD, 4=CNV, 5=CSC, 6=RVO
    dis_list = ['NORMAL', 'OTHERS', 'DR', 'AMD', 'CNV', 'CSC', 'RVO'] #set(datas["Disease"])
    dis_map = {elem:index for index,elem in enumerate(dis_list)} 
    datas['Label'] = datas['Disease'].map(dis_map) #map to the label, 6-normal, 5-others, else-disease
    case_id = datas['ID'].unique().tolist()
    print('Number of Case: {}'.format(len(case_id)))
    case_id_te = random.sample(case_id, int(0.2*len(case_id))) #testset
    case_id_tr = list(set(case_id).difference(set(case_id_te))) #trainset
    datas_tr = pd.DataFrame(columns = datas.columns.values.tolist())
    for case_id in case_id_tr:
        datas_tr = datas_tr.append(datas[datas['ID']==case_id])
    print('Shape of trainset: {}'.format(datas_tr.shape))
    datas_tr.to_csv('/data/pycode/EyeOCT3D/data_oct/OCTA500_Train.txt', index=False, sep=',') #header=False
    datas_te = pd.DataFrame(columns = datas.columns.values.tolist())
    for case_id in case_id_te:
        datas_te = datas_te.append(datas[datas['ID']==case_id])
    print('Shape of trainset: {}'.format(datas_te.shape))
    datas_te.to_csv('/data/pycode/EyeOCT3D/data_oct/OCTA500_Test.txt', index=False, sep=',')

if __name__ == "__main__":

    #split_dataset()

    #generate train files.   
    datasets = get_test_dataloader(batch_size=1, shuffle=False, num_workers=1)
    for batch_idx, (cube, label) in enumerate(datasets):
        print(cube.shape)
        print(label.shape)
 
    #cube = torch.load('/data/tmpexec/OCTA500/test/10187_1.pt')
    #print(cube.shape)
