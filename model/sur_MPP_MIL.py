import copy
import random
import time

import numpy as np
import torch.nn as nn
import  torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GATv2Conv,SAGPooling,global_mean_pool,ASAPooling,global_max_pool,GCNConv,InstanceNorm,GINConv,GENConv,DeepGCNLayer
from torch.nn import   LeakyReLU,LayerNorm

from model.simple_conv import SimapleAvgGraphConv


class MIL(nn.Module):
    def __init__(self, args):
        super(MIL, self).__init__()
        in_classes=args.in_classes
        out_classes = args.out_classes
        drop_out_ratio=args.drop_out_ratio
        self.number_scale = args.number_scale
        self.using_Swin=args.using_Swin
        self.gcn_layer = args.gcn_layer
        self.magnification_scale=args.magnification_scale
        self.l0 = nn.Sequential(
            nn.Linear(in_classes, in_classes//2),
            nn.LeakyReLU(),
            nn.Dropout(drop_out_ratio)
        )
        in_classes=in_classes//2


        self.gnn_convs_diff= torch.nn.ModuleList()

        self.att1 = torch.nn.ModuleList()
        self.att2=torch.nn.ModuleList()
        self.att3=torch.nn.ModuleList()
        self.att_softmax=torch.nn.ModuleList()
        self.att_l1=torch.nn.ModuleList()

        for i in range (self.number_scale):
            self.gnn_convs_diff.append(DeepGCNLayer(
                SimapleAvgGraphConv(in_classes, in_classes),
                LayerNorm(in_classes),
                LeakyReLU(), block='plain', dropout=0, ckpt_grad=0))
            self.att1.append(nn.Sequential(nn.Linear(in_classes*(self.gcn_layer), in_classes*(self.gcn_layer)), nn.Tanh(), nn.Dropout(drop_out_ratio),))
            self.att2.append(nn.Sequential( nn.Linear(in_classes*(self.gcn_layer), in_classes*(self.gcn_layer)),nn.Sigmoid(),nn.Dropout(drop_out_ratio),))
            self.att3.append(nn.Linear(in_classes*(self.gcn_layer) , 1))
            self.att_softmax.append(nn.Softmax(dim=-1))
            self.att_l1.append(nn.Sequential(
                nn.Linear(in_classes*(self.gcn_layer), in_classes*(self.gcn_layer)),
                nn.LeakyReLU(),
                nn.Dropout(drop_out_ratio),
            ))



        self.l_last=nn.Sequential(
                nn.Linear(in_classes*self.number_scale, in_classes*self.number_scale),
                nn.LeakyReLU(),
                nn.Dropout(drop_out_ratio),
            )
        self.l_cla =  nn.Sequential(torch.nn.Linear(in_classes*self.number_scale, out_classes),nn.Sigmoid() )




    def forward(self, x,edge_index_diff,feats_size_list):
        # tim=time.time()
        x = self.l0(x)
        pssz=[]
        for i in range(self.number_scale):
            pssz.append(0)
        if self.using_Swin ==1:
            bag_count_sigle_layer=1
            bag_count_idx=0
            for i in range(self.number_scale):
                for j in range(bag_count_sigle_layer):
                    pssz[i]=pssz[i]+feats_size_list[bag_count_idx]
                    bag_count_idx+=1
                if i<=0:
                    bag_count_sigle_layer=bag_count_sigle_layer*self.magnification_scale*self.magnification_scale
                # bag_count_sigle_layer = bag_count_sigle_layer * self.magnification_scale * self.magnification_scale
        else:
            for i in range(len(feats_size_list)):
                pssz[i]=feats_size_list[i]
        rm_x_count=0
        all_x_count=x.shape[0]
        x_=[]
        # print(pssz)
        for i in range(self.number_scale):
            if pssz[i]<16 and i>0:
                break
            x_.append(torch.split(x, [rm_x_count,pssz[i],all_x_count-rm_x_count-pssz[i]], 0)[1])
            if i != (self.number_scale - 1):
                x = torch.split(x,[rm_x_count, pssz[i] + pssz[i + 1], all_x_count - rm_x_count - pssz[i] - pssz[i + 1]],0)
                xx = x[1]
                edge_index_diff[i] = edge_index_diff[i] - rm_x_count
                xx = self.gnn_convs_diff[i](xx, edge_index_diff[i])
                x = torch.cat((x[0], xx, x[2]))
                edge_index_diff[i] = edge_index_diff[i] + rm_x_count
                rm_x_count=rm_x_count+pssz[i]


        x_v_list=[]
        at_=[]

        for i in range (self.number_scale):
            if pssz[i]<16 and i>0:
                break
            x_sub=x_[i]
            x_sub = self.att_l1[i](x_sub)
            at1 = self.att1[i](x_sub)
            at2 = self.att2[i](x_sub)
            a = at1.mul(at2)
            a = self.att3[i](a)
            a = torch.transpose(a, 0, 1)  # 1*N
            a = self.att_softmax[i](a)
            at_.append(a)
            x_sub = torch.mm(a, x_sub)
            x_v_list.append(x_sub)
        x_v =  torch.cat(x_v_list,-1)
        x_v= F.pad(x_v,(0,self.number_scale*512-x_v.shape[1]))
        x_v=self.l_last(x_v)
        x_v=self.l_cla(x_v)
        # print("INnet:",time.time()-tim)
        return x_v,at_

