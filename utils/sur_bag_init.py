
import pandas as pd
import glob
import os



def sur_get_tcga_luad_bags(args):
    sys_name_list=[]
    luad_list = glob.glob(os.path.join('/data/ly/TCGA_LUAD_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in luad_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_luad_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if name in sys_name_list and name not in WSI_name_list:
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list


def sur_get_tcga_ucec_bags(args):
    sys_name_list=[]
    luad_list = glob.glob(os.path.join('/data/ly/TCGA_UCEC_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in luad_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_ucec_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if name in sys_name_list and name not in WSI_name_list:
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list

def sur_get_tcga_brca_idc_bags(args):
    sys_name_list=[]
    brca_list = glob.glob(os.path.join('/data/ly/TCGA_BRCA_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in brca_list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_brca_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        oncotree=df['oncotree_code'][i]
        if (name in sys_name_list) and (name not in WSI_name_list) :
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list

def sur_get_tcga_blca_bags(args):
    sys_name_list=[]
    list = glob.glob(os.path.join('/data/ly/TCGA_BLCA_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_blca_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if (name in sys_name_list) and (name not in WSI_name_list) :
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list

def sur_get_tcga_gbmlgg_bags(args):
    sys_name_list=[]
    list = glob.glob(os.path.join('/data/ly/TCGA_GBMLGG_Feats', '*'+str(args.patch_size)+'_0_0.csv'))
    for i in list:
        name = i.split(os.sep)[-1].split('_')[0]
        sys_name_list.append(name)

    df = pd.read_csv('file_us/tcga_gbmlgg_all_clean.csv')
    WSI_name_list = []
    sur_time_list = []
    censor_list = []
    bin_sur_time=[0]*30
    for i in range(0, df.shape[0]):
        name=df['slide_id'][i].split('.')[0]
        if (name in sys_name_list) and (name not in WSI_name_list) :
            WSI_name_list.append(name)
            sur_time_list.append(int(df['survival_months'][i]/10))
            bin_sur_time[int(df['survival_months'][i]/10)]+=1
            censor_list.append(int(df['censorship'][i]))
    print('bin sur time:',bin_sur_time)
    return WSI_name_list, sur_time_list,censor_list




