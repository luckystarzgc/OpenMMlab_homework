# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import os

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from mmpretrain import ImageClassificationInferencer

inferencer = ImageClassificationInferencer(
    model='I:/mmpretrain-main/m_cofige.py',  # 自己的config文件
    pretrained='I:/mmpretrain-main/work_dirs/m_cofige/best_accuracy_top1_epoch_11.pth',  # 自己练的权重文件
    device='cuda',  # 用cpu的话可以打'cpu'

)

prediction = inferencer('./1.jpg')[0]
print('Results:', prediction['pred_class'])