# encoding: utf-8
"""
3D UNet
Author: Jason.Fang
Update time: 02/06/2021
"""
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
#define by myself

#https://github.com/qianjinhao/circle-loss/blob/master/circle_loss.py
class CircleLoss(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        
        labels = (labels.cpu().data + 1)
        labels = labels.unsqueeze(1)
        mask = torch.matmul(labels, torch.t(labels))
        #1x1,2x2,3x3,4x4,5x5, 6x6,7x7, 
        mask = mask.eq(1).int() + mask.eq(4).int() + mask.eq(9).int() + \
            mask.eq(16).int() + mask.eq(25).int() + mask.eq(36).int() + mask.eq(47).int() + \
                mask.eq(3*4).int() + mask.eq(3*5).int() + mask.eq(3*6).int() + mask.eq(3*7).int() + \
                    mask.eq(4*5).int() + mask.eq(4*6).int() + mask.eq(4*7).int() + \
                        mask.eq(5*6).int() + mask.eq(5*7).int() + mask.eq(6*7).int()
               

        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

#3DConv Feature Map
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.elu(x)
        return x

class Conv3DNet(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=(2,2,2)):
        super(Conv3DNet, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=(2,2,2), padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
            elif k.startswith("max_pooling"):
                x = op(x)
        return x

# Generalized-Mean (GeM) pooling layer
# https://arxiv.org/pdf/1711.02512.pdf 
class GeMLayer(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMLayer, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def _gem(self, x, p=3, eps=1e-6):
        return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1. / p)
        #return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def forward(self, x):
        return self._gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

#Cross-Slice Attention
class CrossSliceAttention(nn.Module):
    """ Constructs a CSA module.
        Args:k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(CrossSliceAttention, self).__init__()
        self.avg_3dpool = nn.AdaptiveAvgPool3d(1)
        #self.avg_2dpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #cross-channel
        y = self.avg_3dpool(x) 
        y = y.squeeze(-1).squeeze(-1).transpose(-1, -2) #(B, C, 1, 1, 1) -> (B, C, 1)->(B, 1, C)
        y = self.conv(y) #linear projection
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1) #(B, 1, C)-> (B, C, 1) -> (B, C, 1, 1, 1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)# Multi-scale information fusion
        return x

class CT3DIRNet(nn.Module):
    def __init__(self, in_channels, code_size=512, model_depth=4):
        super(CT3DIRNet, self).__init__()
        self.conv3d = Conv3DNet(in_channels=in_channels, model_depth=model_depth)
        self.csa = CrossSliceAttention()
        self.gem = GeMLayer()
        self.fc = nn.Sequential(nn.Linear(512, code_size), nn.Sigmoid()) #for metricl learning
        #self.fc = nn.Sequential(nn.Linear(512, code_size), nn.ReLU()) # for classification

    def forward(self, x):
        x = self.conv3d(x)
        x = self.csa(x)
        x = self.gem(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    #for debug  
    scan =  torch.rand(10, 2, 80, 80, 80).cuda()
    model = CT3DIRNet(in_channels=2, code_size=7).cuda()
    out = model(scan)
    print(out.shape)