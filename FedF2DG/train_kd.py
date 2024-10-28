from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
from itertools import chain
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dst
from st import SoftTarget
from torch.utils.data import DataLoader 
import copy
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils import *

class MyDataset(Dataset):
    def __init__(self,train_data,softlabel):
        self.imgs = train_data.detach().cpu()
        self.imgs.requires_grad = False
        self.softlabels = softlabel.detach().cpu()
        self.softlabels.requires_grad = False

    def __getitem__(self, index):
        data = self.imgs[index]
        softlabel = self.softlabels[index]
        return data , softlabel 

    def __len__(self):
        return len(self.imgs)
        
def local_kd(nets, selected,  global_model, args, clients_data , clients_datatarget, device="cpu"):
    for epoch in range(1,args.kd_epochs+1):
        for idx in selected:
            nets[idx].to(device)
            global_model.to(device)

            nets[idx].eval()
            for param in nets[idx].parameters():
                param.requires_grad = False
        
            criterionKD = SoftTarget(T=1).to(device)
            criterionCls = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(global_model.parameters(),lr = args.kd_lr,weight_decay=args.weight_decay)
            
            train_data = MyDataset(clients_data[idx][1:],clients_datatarget[idx][1:])
            train_loader = DataLoader(dataset=train_data,batch_size=args.bs,shuffle=True,num_workers=4)
            
            st_net = {'snet':global_model, 'tnet':nets[idx]}
            criterions = {'criterionCls':criterionCls, 'criterionKD':criterionKD}
            #训练学生模型
            train(train_loader,st_net,optimizer,criterions,epoch,device)  
            #恢复
            for param in nets[idx].parameters():
                param.requires_grad = True

    return 1

def train(train_loader,nets,optimizer,criterions,epoch,device,test_dl_global,idx):
    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionKD  = criterions['criterionKD']
    tnet.eval()
    snet.train()
    ave_loss=0
    count=0
    for i, (images,label) in enumerate(train_loader):
        images = images.to(device) 
        label = label.to(device)

        optimizer.zero_grad()
        snet.zero_grad()

        s_outputs = snet(images)
        t_outputs = tnet(images)
        kd_loss = criterionKD(s_outputs,t_outputs.detach())
        loss = 1e-3 * kd_loss
        # if i == 0:
        print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
        ave_loss+=loss.data.item()
        count+=1
        loss.backward()
        optimizer.step()

        snet.eval()
        test_acc, conf_matrix = compute_accuracy(snet, test_dl_global, get_confusion_matrix=True, device=device)
        logger.info('>>Epoch %d , idx: %d, Batch: %d Global Model Test round: %f' % (epoch,idx,i,test_acc))
        snet.train()
    logger.info('>>Epoch %d , idx: %d, average loss: %f' % (epoch,idx,ave_loss/count))

def kd(global_model, args, total_data , total_softlabel, device,test_dl_global,global_test_acc=0):
    best_model = copy.deepcopy(global_model)
    best_acc = global_test_acc
    # best_model = None
    # best_acc = 0
    for epoch in range(1,args.kd_epochs+1):
        # optimizer = torch.optim.Adam(global_model.parameters(), lr=args.kd_lr)
        optimizer = torch.optim.SGD(global_model.parameters(),
									lr = args.kd_lr, 
									momentum = args.momentum, 
									weight_decay = args.weight_decay,
                                    nesterov=True
                                  )
        train_data = MyDataset(total_data,total_softlabel)
        train_loader = DataLoader(dataset=train_data,batch_size= 256 ,shuffle=True,num_workers=4)
 
        criterionCls = torch.nn.CrossEntropyLoss().to(device)
        criterionKD = SoftTarget(T=3).to(device)

        global_model.train()
        ave_loss=0
        count=0
        for i, (images,softlabels) in enumerate(train_loader):
            images = images.to(device) 
            softlabels = softlabels.to(device)
            
            optimizer.zero_grad()
            global_model.zero_grad()
            s_output = global_model(images)
            
            kd_loss = criterionKD(s_output,softlabels)
            # cls_loss = criterionCls(s_output,labels)
            loss = 1e-3 * kd_loss 
     
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
            ave_loss += loss.data.item()
            count+=1
            loss.backward()
            optimizer.step()

            global_model.eval()
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)
            logger.info('>>Epoch %d , Batch: %d Global Model Test round: %f' % (epoch,i,test_acc))
            global_model.train()
        logger.info('>>Epoch %d , average loss: %f' % (epoch,ave_loss/count)) 
        test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)
        logger.info('>> Epoch: %d, Global Model Test accuracy: %f' % (epoch,test_acc)) 
        if test_acc >= best_acc :
            best_acc = test_acc
            best_model = copy.deepcopy(global_model)
    return best_model