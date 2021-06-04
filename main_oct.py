# encoding: utf-8
"""
Training implementation for CT-3D retrieval
Author: Jason.Fang
Update time: 03/06/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import ndcg_score
#self-defined
from config import *
from utils.logger import get_logger
from data_oct.oct_dataloader import get_train_dataloader, get_test_dataloader
from nets.oct_net import CT3DIRNet, CircleLoss

#command parameters
parser = argparse.ArgumentParser(description='For EyeOCT')
parser.add_argument('--model', type=str, default='EyeOCT', help='EyeOCT')
args = parser.parse_args()
#config
os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
logger = get_logger(config['log_path'])

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
    dataloader_test = get_test_dataloader(batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if args.model == 'EyeOCT':
        model = CT3DIRNet(in_channels=2, code_size=config['CODE_SIZE'])
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    else: 
        print('No required model')
        return #over
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    #define loss function
    criterion = CircleLoss(scale=8).cuda() #nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    loss_min = float('inf')
    
    for epoch in range(config['MAX_EPOCHS']):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , config['MAX_EPOCHS']))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (cube, label) in enumerate(dataloader_train):
                #forward
                var_cube = torch.autograd.Variable(cube).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_out = model(var_cube)
                # backward and update parameters
                optimizer_model.zero_grad()
                loss_tensor = criterion.forward(var_out, var_label)
                loss_tensor.backward()
                optimizer_model.step()
                #show 
                loss_train.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(loss_train) ))

        #test
        model.eval()
        loss_test = []
        with torch.autograd.no_grad():
            for batch_idx, (cube, label)  in enumerate(dataloader_test):
                #forward
                var_cube = torch.autograd.Variable(cube).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_out = model(var_cube)
                loss_tensor = criterion.forward(var_out, var_label)
                loss_test.append(loss_tensor.item())
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        print("\r Eopch: %5d test loss = %.6f" % (epoch + 1, np.mean(loss_test) ))

        # save checkpoint
        if loss_min > np.mean(loss_test):
            loss_min = np.mean(loss_test)
            torch.save(model.module.state_dict(), config['CKPT_PATH'] + args.model + '_best.pkl') #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))


def Test():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=5, shuffle=False, num_workers=8) #config['BATCH_SIZE']
    dataloader_test = get_test_dataloader(batch_size=5, shuffle=False, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    if args.model == 'EyeOCT':
        model = CT3DIRNet(in_channels=2, code_size=config['CODE_SIZE']).cuda()
        CKPT_PATH = config['CKPT_PATH'] + args.model + '_best.pkl'
        if os.path.exists(CKPT_PATH):
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint) #strict=False
            print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    else: 
        print('No required model')
        return #over
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    print('********************Build feature database!********************')
    tr_label = torch.FloatTensor().cuda()
    tr_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (cube, label) in enumerate(dataloader_train):
            var_cube = torch.autograd.Variable(cube).cuda()
            var_out = model(var_cube)
            tr_feat = torch.cat((tr_feat, var_out.data), 0)
            tr_label = torch.cat((tr_label, label.cuda()), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (cube, label) in enumerate(dataloader_test):
            var_cube = torch.autograd.Variable(cube).cuda()
            var_out = model(var_cube)
            te_feat = torch.cat((te_feat, var_out.data), 0)
            te_label = torch.cat((te_label, label.cuda()), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.cpu().numpy(), tr_feat.cpu().numpy())
    te_label = te_label.cpu().numpy()
    tr_label = tr_label.cpu().numpy()

    for topk in [5]: #[5,10,20,50]:
        mHRs_avg = []
        mAPs_avg = []
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = te_label[i]#te_label[i,:][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = tr_label[j]#tr_label[j,:][0]
                if te_idx == tr_idx:  #hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos)
                elif te_idx in [2., 3., 4., 5., 6.] and tr_idx in [2., 3., 4., 5., 6.]: #disease hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs_avg.append(0)
            mHRs_avg.append(num_pos/rank_pos)

            sys.stdout.write('\r test set process: = {}'.format(i+1))
            sys.stdout.flush()

        #Hit ratio
        logger.info("HR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        #average precision
        logger.info("AP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()
