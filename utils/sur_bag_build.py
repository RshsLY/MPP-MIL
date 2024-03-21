import copy
import math
import os
import time

import pandas
import numpy as np
import torch
def get_bag(args,WSI_name, sur_time,censor,data_map):

    csv_path = ""
    if args.dataset == 'TCGA_LUAD':
        csv_path = os.path.join('/data/ly/TCGA_LUAD_Feats', WSI_name + '_'+str(args.patch_size)+'_0_0' + '.csv')
    if args.dataset == 'TCGA_UCEC':
        csv_path = os.path.join('/data/ly/TCGA_UCEC_Feats', WSI_name + '_'+str(args.patch_size)+'_0_0' + '.csv')
    if args.dataset == 'TCGA_BRCA':
        csv_path = os.path.join('/data/ly/TCGA_BRCA_Feats', WSI_name + '_' + str(args.patch_size) + '_0_0' + '.csv')
    if args.dataset == 'TCGA_BLCA':
        csv_path = os.path.join('/data/ly/TCGA_BLCA_Feats', WSI_name + '_' + str(args.patch_size) + '_0_0' + '.csv')
    if args.dataset == 'TCGA_GBMLGG':
        csv_path = os.path.join('/data/ly/TCGA_GBMLGG_Feats', WSI_name + '_' + str(args.patch_size) + '_0_0' + '.csv')


    WSI_name=WSI_name+"_scale"+str(args.number_scale)

    if data_map.__contains__(WSI_name)==False:
        #tim=time.time()
        feats_mp = {}
        feats_list = []
        feats_info=[]
        patch_size_list=[args.patch_size]
        for i in range(args.number_scale-1):
            patch_size_list.append(patch_size_list[patch_size_list.__len__()-1]*args.magnification_scale)
        feats_size_list=[]
        feats_count=0
        min_ps=patch_size_list[0]
        max_row=0
        max_col=0
        df = pandas.read_csv(csv_path)
        for i in range(0, df.shape[1]):
            pos_str = df.columns[i]
            top_row = int(pos_str.split(',')[0])
            top_col = int(pos_str.split(',')[1])
            max_row=max(max_row,top_row+min_ps+1)
            max_col=max(max_col,top_col+min_ps+1)
            feats_list.append(df[df.columns[i]].values)
            feats_info.append((min_ps,top_row,top_col))
            feats_mp[(min_ps, 0, 0, top_row, top_col)] = feats_count
            feats_count = feats_count + 1
        feats_size_list.append(feats_count)

        for now_ps in patch_size_list:
            if now_ps==min_ps:
                continue
            for start_row in range(-now_ps + now_ps//args.magnification_scale, 1, now_ps//args.magnification_scale):
                for start_col in range(-now_ps + now_ps//args.magnification_scale, 1, now_ps//args.magnification_scale):
                    if args.using_Swin == 0  and (start_col!=0 or start_row!=0):
                        continue
                    feats_count_now=0
                    for top_row in range(start_row, max_row, now_ps):
                        for top_col in range(start_col, max_col, now_ps):
                            flag_sub=0
                            pre_scale=now_ps // args.magnification_scale
                            for SR in range( -pre_scale+ pre_scale//args.magnification_scale, 1, pre_scale//args.magnification_scale):
                                for SC in range(-pre_scale + pre_scale // args.magnification_scale, 1,pre_scale // args.magnification_scale):

                                    if SR==-2 and SC==-2:
                                        SR=0
                                        SC=0
                                    for TR in range(top_row, top_row + now_ps, now_ps//args.magnification_scale):
                                        for TC in range(top_col, top_col + now_ps, now_ps//args.magnification_scale):
                                            idx = (now_ps // args.magnification_scale, SR, SC, TR, TC)
                                            if  flag_sub==0 and feats_mp.__contains__(idx)  \
                                                    and TR+now_ps//args.magnification_scale <= top_row + now_ps and TC+now_ps//args.magnification_scale <= top_col +now_ps:
                                                flag_sub+=1
                            #print(now_ps,'-',flag_sub)
                            if flag_sub>=1:
                                feats_list.append([0]*1024)
                                feats_info.append((now_ps,top_row,top_col))
                                feats_mp[(now_ps, start_row, start_col, top_row, top_col)] = feats_count
                                feats_count=feats_count+1
                                feats_count_now = feats_count_now + 1
                    feats_size_list.append(feats_count_now)


        edge_index_diff=[]
        for i in range(args.number_scale):
            edge_index_diff.append([[],[]])
        for i in feats_mp:
            now_ps = i[0]
            start_row = i[1]
            start_col = i[2]
            top_row = i[3]
            top_col = i[4]

            pre_scale = now_ps // args.magnification_scale
            for SR in range(-pre_scale + pre_scale // args.magnification_scale, 1,pre_scale // args.magnification_scale):
                for SC in range(-pre_scale + pre_scale // args.magnification_scale, 1,pre_scale // args.magnification_scale):
                    if SR == -2 and SC == -2:
                        SR = 0
                        SC = 0
                    for TR in range(top_row, top_row + now_ps, now_ps//args.magnification_scale):
                        for TC in range(top_col, top_col + now_ps, now_ps//args.magnification_scale):
                            idx = (now_ps // args.magnification_scale, SR, SC, TR, TC)
                            if  feats_mp.__contains__(idx) and TR+now_ps//args.magnification_scale <= top_row + now_ps and TC+now_ps//args.magnification_scale <= top_col + now_ps:
                                for sz_idx in range(patch_size_list.__len__()):
                                    if patch_size_list[sz_idx] == now_ps:
                                        edge_index_diff[sz_idx-1][1].append(feats_mp[idx])
                                        edge_index_diff[sz_idx-1][0].append(feats_mp[i])
                                        edge_index_diff[sz_idx-1][0].append(feats_mp[idx])
                                        edge_index_diff[sz_idx-1][1].append(feats_mp[i])
        feats_list = np.array(feats_list)
        feats_list = torch.from_numpy(feats_list)
        feats_list = feats_list.to(torch.float32)

        for idx in range(edge_index_diff.__len__()):
            edge_index_diff[idx] = np.array(edge_index_diff[idx])
            edge_index_diff[idx] = torch.from_numpy(edge_index_diff[idx])
            edge_index_diff[idx] = edge_index_diff[idx].to(torch.long)
        sur_time = np.array(sur_time)
        sur_time = torch.from_numpy(sur_time)
        sur_time = sur_time.to(torch.long)
        censor = np.array(censor)
        censor = torch.from_numpy(censor)
        censor = censor.to(torch.long)
        data_map[WSI_name]={"feats_list":feats_list,"sur_time":sur_time,"censor":censor,"edge_index_diff":edge_index_diff,
                            "feats_size_list":feats_size_list,"feats_info":feats_info}
        #print("building:", time.time() - tim)
    #tim=time.time()
    feats_list=copy.deepcopy(data_map[WSI_name]["feats_list"]).cuda()
    sur_time=copy.deepcopy(data_map[WSI_name]["sur_time"]).cuda()
    censor=copy.deepcopy(data_map[WSI_name]["censor"]).cuda()

    edge_index_diff = copy.deepcopy(data_map[WSI_name]["edge_index_diff"])

    for i in range(len(edge_index_diff)):
        edge_index_diff[i] = edge_index_diff[i].cuda()
    feats_size_list=copy.deepcopy(data_map[WSI_name]["feats_size_list"])
    feats_info=copy.deepcopy(data_map[WSI_name]["feats_info"])
    #print("mapping:",time.time()-tim)
    return feats_list,sur_time,censor,edge_index_diff,feats_size_list,feats_info



