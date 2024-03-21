from PIL import ImageDraw
from matplotlib import gridspec
import  pandas as pd


def fun(args):
    feats_list, sur_time, censor, edge_index_diff, feats_size_list, feats_info = \
        utils.sur_bag_build.get_bag(args, args.WSI_name, -1, -1, {})
    prediction_list, at_ = model(feats_list, edge_index_diff, feats_size_list)

    at_c = (at_[0] - torch.min(at_[0][0])) / (torch.max(at_[0][0]) - torch.min(at_[0][0])) * 255
    for i in range(1, len(at_)):
        at_[i] = (at_[i] - torch.min(at_[i][0])) / (torch.max(at_[i][0]) - torch.min(at_[i][0])) * 255
        at_c = torch.cat((at_c, at_[i]), dim=-1)

    at_c = at_c.cpu().detach().numpy()

    heat_map = np.zeros([pil_img.size[1], pil_img.size[0]])
    is_color = np.zeros([pil_img.size[1], pil_img.size[0]])
    if tif.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER) == '40':
        print("40!!!!!!!!!!!!!")
    now_ps = args.patch_size
    beishu = 1
    if args.layer_select!=0:
        beishu *= args.magnification_scale
        beishu *= args.magnification_scale
    for i in range(args.layer_select):
        now_ps = now_ps * args.magnification_scale
    for idx, (ps, tr, tc) in enumerate(feats_info):
        if idx>=len(at_c[0]):
            continue
        if ps != now_ps: continue
        ps1 = ps // args.down
        tr1 = tr // args.down
        tc1 = tc // args.down
        if tif.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER) == '40':
            # print("40!!!")
            ps1 *= 2
            tr1 *= 2
            tc1 *= 2
        for i in range(tr1, ps1 + tr1):
            for j in range(tc1, ps1 + tc1):
                if i>=0 and i < pil_img.size[1] and j>=0 and j < pil_img.size[0]:
                    if args.using_Swin == 1:
                        heat_map[i][j] += at_c[0][idx]/beishu
                    else:
                        heat_map[i][j] += at_c[0][idx]
                    is_color[i][j] = 1
    # heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map)- np.min(heat_map))
    heat_map = heat_map.astype(np.uint8)
    heat_map1 = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)  # BGR
    heat_map1 = cv2.cvtColor(heat_map1, cv2.COLOR_BGR2RGB)
    for i in range(len(is_color)):
        for j in range(len(is_color[0])):
            if is_color[i][j] == 0:
                heat_map1[i][j][0] = 255
                heat_map1[i][j][1] = 255
                heat_map1[i][j][2] = 255
    heat_map1 = heat_map1.astype(np.uint8)
    heat_map1 = PIL.Image.fromarray(heat_map1)
    # heat_map1.show()
    final_img = PIL.Image.blend(pil_img, heat_map1, 0.5)
    final_img.save("hm0305\\" + args.WSI_name +"_final"+ str(args.layer_select) + '.png')
    return final_img

import os
import openslide
import  argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str,help="path for saved model")
parser.add_argument("--tif_path",type=str,default="svs/TCGA-D1-A15Z-01Z-00-DX1.svs")
parser.add_argument("--WSI_name",type=str,default="TCGA-D1-A15Z-01Z-00-DX1")

parser.add_argument("--layer_select", type=int,default=0)
parser.add_argument("--down", type=int,default=64)

parser.add_argument("--patch_size", type=int,            default=512,               help="patch_size to use")
parser.add_argument('--gpu_index', type=int,             default=6,                 help='GPU ID(s)')
parser.add_argument("--dataset", type=str,               default="TCGA_UCEC",       help="Database to use")
parser.add_argument("--model", type=str,                 default="sur_MPP_MIL", )
parser.add_argument("--in_classes", type=int,            default=1024,              help="Feature size of each patch")
parser.add_argument("--out_classes", type=int,           default=30,                help="Number of classes,UCEC,LUAD=25 BLCA=30")

parser.add_argument("--number_scale", type=int,          default=7,                 help="")
parser.add_argument("--using_Swin",type=int,             default=1,                 help="[0,1]")
parser.add_argument("--magnification_scale",             type=int, default=3, help="")
parser.add_argument("--gcn_layer", type=int,             default=1)
parser.add_argument("--divide_seed", type=int,           default=2023,              help="Data division seed")
# ------------------
parser.add_argument("--drop_out_ratio", type=float,      default=0.25,               help="Drop_out_ratio")

args, _ = parser.parse_known_args()

import os

gpu_ids = tuple((args.gpu_index,))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
import PIL.Image
import numpy as np

import model.sur_MPP_MIL as mil
import torch
import utils.sur_bag_build
import cv2

tif = openslide.OpenSlide(args.tif_path)
pil_img = tif.get_thumbnail((tif.level_dimensions[0][0] // args.down, tif.level_dimensions[0][1] // args.down))
pil_img1 = tif.get_thumbnail((tif.level_dimensions[0][0] // args.down, tif.level_dimensions[0][1] // args.down))
pil_img1.save('hm0305\\' + args.WSI_name + '_ORI' + '.png')


list=[]
print(tif.level_dimensions[0][0], tif.level_dimensions[0][1])
model = mil.MIL(args)
model = model.cpu()
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.eval()
for i in range(0,7):
    args.layer_select=i
    fimg=fun(args)
    list.append(fimg)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 2, width_ratios=[1, 1 ], height_ratios=[1, 1,1,1])


for i, img in enumerate(list):
    ax = plt.subplot(gs[i])
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)

plt.show()



