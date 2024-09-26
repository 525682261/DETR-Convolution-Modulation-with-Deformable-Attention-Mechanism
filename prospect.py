import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

import matplotlib.pyplot as plt
from PIL import Image
import datasets.transforms as T
import math
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F

model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
model_checkpoint_path = "/22085400506/DyConv1d-dis-lf2/logs/DINO/R50-MS4-dis-Dyconv1d-lf2/checkpoint0011.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# image = Image.open("./figs/idea.jpg").convert("RGB") # load image

# # transform images
# transform = T.Compose([
#     T.RandomResize([800], max_size=1333),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# image, _ = transform(image, None)

# load coco names
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}

args.dataset_file = 'coco'
args.coco_path = "/22085400506/dataset/coco" # the path of coco
args.fix_size = False

dataset_val = build_dataset(image_set='val', args=args)

image, targets = dataset_val[71]

c, h, w = image.shape

# predict images
output = model.cuda()(image[None].cuda())

enc_list = output['memory_list']
enc = enc_list[3]
memory = output['memory']
level_start_index = output['level_start_index']
spatial_shapes = output['spatial_shapes']

# memory0 = memory[:, :level_start_index[1], :]
# memory1 = memory[:, level_start_index[1]:level_start_index[2], :]
# memory2 = memory[:, level_start_index[2]:level_start_index[3], :]
# memory3 = memory[:, level_start_index[3]:, :]
enc_0 = enc[:, :level_start_index[1], :]
enc_1 = enc[:, level_start_index[1]:level_start_index[2], :]
enc_2 = enc[:, level_start_index[2]:level_start_index[3], :]
enc_3 = enc[:, level_start_index[3]:, :]

# f0 = memory0.reshape(memory.shape[0], spatial_shapes[0][0], spatial_shapes[0][1], memory.shape[2])
# f1 = memory1.reshape(memory.shape[0], spatial_shapes[1][0], spatial_shapes[1][1], memory.shape[2])
# f2 = memory2.reshape(memory.shape[0], spatial_shapes[2][0], spatial_shapes[2][1], memory.shape[2])
# f3 = memory3.reshape(memory.shape[0], spatial_shapes[3][0], spatial_shapes[3][1], memory.shape[2])

enc_f0 = enc_0.reshape(memory.shape[0], spatial_shapes[0][0], spatial_shapes[0][1], memory.shape[2])
enc_f1 = enc_1.reshape(memory.shape[0], spatial_shapes[1][0], spatial_shapes[1][1], memory.shape[2])
enc_f2 = enc_2.reshape(memory.shape[0], spatial_shapes[2][0], spatial_shapes[2][1], memory.shape[2])
enc_f3 = enc_3.reshape(memory.shape[0], spatial_shapes[3][0], spatial_shapes[3][1], memory.shape[2])

# f0 = np.array(f0, dtype=float)
# f0 = torch.from_numpy(f0)

# f0 = f0.permute(0, 3, 1, 2)
# f1 = f1.permute(0, 3, 1, 2)
# f2 = f2.permute(0, 3, 1, 2)
# f3 = f3.permute(0, 3, 1, 2)

enc_f0 = enc_f0.permute(0, 3, 1, 2)
enc_f1 = enc_f1.permute(0, 3, 1, 2)
enc_f2 = enc_f2.permute(0, 3, 1, 2)
enc_f3 = enc_f3.permute(0, 3, 1, 2)

# f0_ = F.interpolate(f0, scale_factor=(w/f0.shape[3], h/f0.shape[2]), mode='bilinear')
# f1_ = F.interpolate(f1, scale_factor=(w/f1.shape[3], h/f1.shape[2]), mode='bilinear')
# f2_ = F.interpolate(f2, scale_factor=(w/f2.shape[3], h/f2.shape[2]), mode='bilinear')
# f3_ = F.interpolate(f3, scale_factor=(w/f3.shape[3], h/f3.shape[2]), mode='bilinear')

enc_f0_ = F.interpolate(enc_f0, scale_factor=(w/enc_f0.shape[3], h/enc_f0.shape[2]), mode='bilinear')
enc_f1_ = F.interpolate(enc_f1, scale_factor=(w/enc_f1.shape[3], h/enc_f1.shape[2]), mode='bilinear')
enc_f2_ = F.interpolate(enc_f2, scale_factor=(w/enc_f2.shape[3], h/enc_f2.shape[2]), mode='bilinear')
enc_f3_ = F.interpolate(enc_f3, scale_factor=(w/enc_f3.shape[3], h/enc_f3.shape[2]), mode='bilinear')

# f0_ln = torch.nn.functional.normalize(f0_, p=1, dim=1)
# f1_ln = torch.nn.functional.normalize(f1_, p=1, dim=1)
# f2_ln = torch.nn.functional.normalize(f2_, p=1, dim=1)
# f3_ln = torch.nn.functional.normalize(f3_, p=1, dim=1)

enc_f0_ln = torch.nn.functional.normalize(enc_f0_, p=1, dim=1)
enc_f1_ln = torch.nn.functional.normalize(enc_f1_, p=1, dim=1)
enc_f2_ln = torch.nn.functional.normalize(enc_f2_, p=1, dim=1)
enc_f3_ln = torch.nn.functional.normalize(enc_f3_, p=1, dim=1)

# f0_ln = f0_ln.to('cpu')
# f1_ln = f1_ln.to('cpu')
# f2_ln = f2_ln.to('cpu')
# f3_ln = f3_ln.to('cpu')

enc_f0_ln = enc_f0_ln.to('cpu')
enc_f1_ln = enc_f1_ln.to('cpu')
enc_f2_ln = enc_f2_ln.to('cpu')
enc_f3_ln = enc_f3_ln.to('cpu')

# f0_ln_av = torch.mean(f0_ln, 1).squeeze(0).detach().numpy()
# f1_ln_av = torch.mean(f1_ln, 1).squeeze(0).detach().numpy()
# f2_ln_av = torch.mean(f2_ln, 1).squeeze(0).detach().numpy()
# f3_ln_av = torch.mean(f3_ln, 1).squeeze(0).detach().numpy()

enc_f0_ln_av = torch.mean(enc_f0_ln, 1).squeeze(0).detach().numpy()
enc_f1_ln_av = torch.mean(enc_f1_ln, 1).squeeze(0).detach().numpy()
enc_f2_ln_av = torch.mean(enc_f2_ln, 1).squeeze(0).detach().numpy()
enc_f3_ln_av = torch.mean(enc_f3_ln, 1).squeeze(0).detach().numpy()

# f0_ln_av_m = np.max(f0_ln_av)
# f1_ln_av_m = np.max(f1_ln_av)
# f2_ln_av_m = np.max(f2_ln_av)
# f3_ln_av_m = np.min(f3_ln_av)

enc_f0_ln_av_m = np.max(enc_f0_ln_av)
enc_f1_ln_av_m = np.max(enc_f1_ln_av)
enc_f2_ln_av_m = np.max(enc_f2_ln_av)
enc_f3_ln_av_m = np.min(enc_f3_ln_av)

# f0_ln_av = (f0_ln_av / f0_ln_av_m) - 1
# f1_ln_av = (f1_ln_av / f1_ln_av_m) - 1
# f2_ln_av = (f2_ln_av / f2_ln_av_m) - 1
# f3_ln_av = (f3_ln_av / f3_ln_av_m)

enc_f0_ln_av = (enc_f0_ln_av / enc_f0_ln_av_m) - 1
enc_f1_ln_av = (enc_f1_ln_av / enc_f1_ln_av_m) - 1
enc_f2_ln_av = (enc_f2_ln_av / enc_f2_ln_av_m) - 1
enc_f3_ln_av = (enc_f3_ln_av / enc_f3_ln_av_m)

# f0_ln_av = (f0_ln_av / f0_ln_av_m)
# f1_ln_av = (f1_ln_av / f1_ln_av_m)
# f2_ln_av = f2_ln_av / f2_ln_av_m
# f3_ln_av = f3_ln_av / f3_ln_av_m

# for i in range(f0_ln_av.shape[0]):
#     for j in range(f0_ln_av.shape[1]):
#         f0_ln_av[i][j] = math.exp(f0_ln_av[i][j])
#
# for i in range(f1_ln_av.shape[0]):
#     for j in range(f1_ln_av.shape[1]):
#         f1_ln_av[i][j] = math.exp(f1_ln_av[i][j])
#
# for i in range(f2_ln_av.shape[0]):
#     for j in range(f2_ln_av.shape[1]):
#         f2_ln_av[i][j] = math.exp(f2_ln_av[i][j])
#
# for i in range(f3_ln_av.shape[0]):
#     for j in range(f3_ln_av.shape[1]):
#         f3_ln_av[i][j] = math.exp(f3_ln_av[i][j])

# for i in range(enc_f0_ln_av.shape[0]):
#     for j in range(enc_f0_ln_av.shape[1]):
#         enc_f0_ln_av[i][j] = math.exp(enc_f0_ln_av[i][j])
#
# for i in range(enc_f1_ln_av.shape[0]):
#     for j in range(enc_f1_ln_av.shape[1]):
#         enc_f1_ln_av[i][j] = math.exp(enc_f1_ln_av[i][j])
#
# for i in range(enc_f2_ln_av.shape[0]):
#     for j in range(enc_f2_ln_av.shape[1]):
#         enc_f2_ln_av[i][j] = math.exp(enc_f2_ln_av[i][j])
#
# for i in range(enc_f3_ln_av.shape[0]):
#     for j in range(enc_f3_ln_av.shape[1]):
#         enc_f3_ln_av[i][j] = math.exp(enc_f3_ln_av[i][j])

# f0_ln_r = np.zeros([h, w])
# f1_ln_r = np.zeros([h, w])
# f2_ln_r = np.zeros([h, w])
# f3_ln_r = np.zeros([h, w])

enc_f0_ln_r = np.zeros([h, w])
enc_f1_ln_r = np.zeros([h, w])
enc_f2_ln_r = np.zeros([h, w])
enc_f3_ln_r = np.zeros([h, w])

#71
# f0_ln_r[:798, :] = f0_ln_av[:, :1062]
# f1_ln_r[:792, :] = f1_ln_av[:, :1062]
# f2_ln_r[:780, :] = f2_ln_av[:, :1062]
# f3_ln_r[:, :1046] = f3_ln_av[:800, :]

enc_f0_ln_r[:798, :] = enc_f0_ln_av[:, :1062]
enc_f1_ln_r[:792, :] = enc_f1_ln_av[:, :1062]
enc_f2_ln_r[:780, :] = enc_f2_ln_av[:, :1062]
enc_f3_ln_r[:, :1046] = enc_f3_ln_av[:800, :]

# #65
# f0_ln_r[:799,:] = f0_ln_av[:,:1199]
# f1_ln_r[:799,:] = f1_ln_av[:,:1199]
# f2_ln_r[:788,:] = f2_ln_av[:,:1199]
# f3_ln_r[:,:1169] = f3_ln_av[:800,:]

#1
# f0_ln_r[:,:,:,:793] = f0_ln[:,:,:873,:]
# f1_ln_r[:,:,:,:793] = f1_ln[:,:,:873,:]
# f2_ln_r[:,:,:,:779] = f2_ln[:,:,:873,:]
# f3_ln_r[:,:,:861,:] = f3_ln[:,:,:,:800]

# f = (f0_ln_r + f1_ln_r + f2_ln_r + f3_ln_r) / 4

enc_f = (enc_f0_ln_r + enc_f1_ln_r + enc_f2_ln_r + enc_f3_ln_r) / 4

# num_b = 0
# num_f = 0
#
# for i in range(f.shape[0]):
#     for j in range(f.shape[1]):
#         if f[i][j] >= 0.5:
#             num_f += 1
#         else:
#             num_b += 1

f_iof = np.zeros((h, w))
f_iob = np.ones((h, w))
# f_iof_ = np.zeros([h, w])
# f_iob_ = f.copy()

# enc_f_iof = np.zeros((h, w))
# enc_f_iob = np.ones((h, w))
enc_f_iof_ = np.zeros([h, w])
enc_f_iob_ = enc_f.copy()

b_num = targets['boxes'].shape[0]
for i in range(b_num):
    boxes = []
    box = targets['boxes'][i]
    unnormbbox = box * torch.Tensor([w, h, w, h])
    unnormbbox[:2] -= unnormbbox[2:] / 2
    [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
    boxes.append([bbox_x, bbox_y, bbox_w, bbox_h])
    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
             [bbox_x + bbox_w, bbox_y]]
    np_poly = np.array(poly).reshape((4, 2))
    np_poly = np.ceil(np_poly)
    np_poly = np_poly.astype(np.int16)
    f_iof[np_poly[0][1]:np_poly[1][1], np_poly[0][0]:np_poly[2][0]] = 1.0
    f_iob[np_poly[0][1]:np_poly[1][1], np_poly[0][0]:np_poly[2][0]] = 0.0
    # f_iof_[np_poly[0][1]:np_poly[1][1], np_poly[0][0]:np_poly[2][0]] = f[np_poly[0][1]:np_poly[1][1],
    #                                                                        np_poly[0][0]:np_poly[2][0]]
    # f_iob_[np_poly[0][1]:np_poly[1][1], np_poly[0][0]:np_poly[2][0]] = 0.0
    enc_f_iof_[np_poly[0][1]:np_poly[1][1], np_poly[0][0]:np_poly[2][0]] = enc_f[np_poly[0][1]:np_poly[1][1],
                                                                       np_poly[0][0]:np_poly[2][0]]
    enc_f_iob_[np_poly[0][1]:np_poly[1][1], np_poly[0][0]:np_poly[2][0]] = 0.0

# box1 = targets['boxes'][0]
# box2 = targets['boxes'][1]
# boxes1 = []
# boxes2 = []
#
# unnormbbox1 = box1 * torch.Tensor([w,h,w,h])
# unnormbbox1[:2] -= unnormbbox1[2:] / 2
# [bbox1_x, bbox1_y, bbox1_w, bbox1_h] = unnormbbox1.tolist()
# boxes1.append([bbox1_x, bbox1_y, bbox1_w, bbox1_h])
# poly1 = [[bbox1_x, bbox1_y], [bbox1_x, bbox1_y+bbox1_h], [bbox1_x+bbox1_w, bbox1_y+bbox1_h], [bbox1_x+bbox1_w, bbox1_y]]
# np_poly1 = np.array(poly1).reshape((4,2))
# np_poly1 = np.ceil(np_poly1)
# np_poly1 = np_poly1.astype(np.int16)
#
# unnormbbox2 = box2 * torch.Tensor([w,h,w,h])
# unnormbbox2[:2] -= unnormbbox2[2:] / 2
# [bbox2_x, bbox2_y, bbox2_w, bbox2_h] = unnormbbox2.tolist()
# boxes2.append([bbox2_x, bbox2_y, bbox2_w, bbox2_h])
# poly2 = [[bbox2_x, bbox2_y], [bbox2_x, bbox2_y+bbox2_h], [bbox2_x+bbox2_w, bbox2_y+bbox2_h], [bbox2_x+bbox2_w, bbox2_y]]
# np_poly2 = np.array(poly2).reshape((4,2))
# np_poly2 = np.ceil(np_poly2)
# np_poly2 = np_poly2.astype(np.int16)
#
# f_iof[np_poly1[0][0]:np_poly1[2][0], np_poly1[0][1]:np_poly1[1][1]] = 1.0
# f_iof[np_poly2[0][0]:np_poly2[2][0], np_poly2[0][1]:np_poly2[1][1]] = 1.0
# f_iob[np_poly1[0][0]:np_poly1[2][0], np_poly1[0][1]:np_poly1[1][1]] = 0.0
# f_iob[np_poly2[0][0]:np_poly2[2][0], np_poly2[0][1]:np_poly2[1][1]] = 0.0

f_iof_sum = np.sum(f_iof)
f_iob_sum = np.sum(f_iob)

# f_iof_ = np.zeros([h, w])
# f_iob_ = f
#
# f_iof_[np_poly1[0][1]:np_poly1[1][1], np_poly1[0][0]:np_poly1[2][0]] = f[np_poly1[0][1]:np_poly1[1][1], np_poly1[0][0]:np_poly1[2][0]]
# f_iof_[np_poly2[0][1]:np_poly2[1][1], np_poly2[0][0]:np_poly2[2][0]] = f[np_poly2[0][1]:np_poly2[1][1], np_poly2[0][0]:np_poly2[2][0]]
#
# f_iob_[np_poly1[0][1]:np_poly1[1][1], np_poly1[0][0]:np_poly1[2][0]] = 0.0
# f_iob_[np_poly2[0][1]:np_poly2[1][1], np_poly2[0][0]:np_poly2[2][0]] = 0.0




S = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# IOF_list = [0,0,0,0,0,0,0,0,0]
# IOB_list = [0,0,0,0,0,0,0,0,0]

enc_IOF_list = [0,0,0,0,0,0,0,0,0]
enc_IOB_list = [0,0,0,0,0,0,0,0,0]

# for h in range(len(S)):
#     f_iof_sum_ = 0
#     for i in range(f_iof_.shape[0]):
#         for j in range(f_iof_.shape[1]):
#             if f_iof_[i][j] > S[h]:
#                 f_iof_sum_ += f_iof_[i][j]
#     IOF = f_iof_sum_ / f_iof_sum
#     IOF_list[h] = IOF

for h in range(len(S)):
    enc_f_iof_sum_ = 0
    for i in range(enc_f_iof_.shape[0]):
        for j in range(enc_f_iof_.shape[1]):
            if enc_f_iof_[i][j] > S[h]:
                enc_f_iof_sum_ += enc_f_iof_[i][j]
    enc_IOF = enc_f_iof_sum_ / f_iof_sum
    enc_IOF_list[h] = enc_IOF

# for i in range(f_iof_.shape[0]):
#     for j in range(f_iof_.shape[1]):
#         if f_iof_[i][j] >= S:
#             f_iof_sum_ += f_iof_[i][j]
# for h in range(len(S)):
#     f_iob_sum_ = 0
#     for i in range(f_iob_.shape[0]):
#         for j in range(f_iob_.shape[1]):
#             if f_iob_[i][j] <= S[h]:
#                 f_iob_sum_ += f_iob_[i][j]
#     IOB = f_iob_sum_ / f_iob_sum
#     IOB_list[h] = IOB

for h in range(len(S)):
    enc_f_iob_sum_ = 0
    for i in range(enc_f_iob_.shape[0]):
        for j in range(enc_f_iob_.shape[1]):
            if enc_f_iob_[i][j] <= S[h]:
                enc_f_iob_sum_ += enc_f_iob_[i][j]
    enc_IOB = enc_f_iob_sum_ / f_iob_sum
    enc_IOB_list[h] = enc_IOB
# for i in range(f_iob_.shape[0]):
#     for j in range(f_iob_.shape[1]):
#         if f_iob_[i][j] < S:
#             f_iob_sum_ += f_iob_[i][j]
#
# IOF = f_iof_sum_ / f_iof_sum
# IOB = f_iob_sum_ / f_iob_sum

# print(IOB_list)
# print(IOF_list)

print(enc_IOB_list)
print(enc_IOF_list)

# plt.plot(IOB_list, IOF_list, 'g^')
# plt.show()
