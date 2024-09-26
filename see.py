import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

import matplotlib.pyplot as plt

model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
model_checkpoint_path = "/22085400506/DyConv1d-dis-lf2/logs/DINO/R50-MS4-dis-Dyconv1d-lf2/checkpoint0011.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

from PIL import Image
import datasets.transforms as T
import math

image = Image.open("./figs/idea.jpg").convert("RGB") # load image

# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)

# with open('util/coco_id2name.json') as f:
#     id2name = json.load(f)
#     id2name = {int(k):v for k,v in id2name.items()}
#
#
# args.dataset_file = 'coco'
# args.coco_path = "/22085400506/dataset/coco" # the path of coco
# args.fix_size = False
#
# dataset_val = build_dataset(image_set='val', args=args)
#
# image, targets = dataset_val[10]

# predict images
output = model.cuda()(image[None].cuda())
#获取偏移量 （1,c,8,4,4,2）
sampling_offsets_list = output['sampling_locations_list']
#获取权重（1,c,8,4,4）
attention_weights_list = output['attention_weights_list']
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

#获取最后编码器一层偏移量的信息
sampling_offsets_last = sampling_offsets_list[5]
#获取最后一张特征层的偏移量 （x，y）
sampling_offsets_last1 = sampling_offsets_last[:, 19700:, :, :, :]
sampling_offsets_last1 = sampling_offsets_last1.to('cpu')
sampling_offsets_last1 = sampling_offsets_last1.detach().numpy()
sampling_offsets_last1 = sampling_offsets_last1[:, :, 0, 3, :, :].reshape(247, 4, 2)
sampling_offsets_last1[:,:,0] = sampling_offsets_last1[:,:,0] * 13
sampling_offsets_last1[:,:,1] = sampling_offsets_last1[:,:,1] * 19
#取绝对值
sampling_offsets_last1 = np.maximum(sampling_offsets_last1, -sampling_offsets_last1)
#取整
sampling_offsets_last1 = np.rint(sampling_offsets_last1)

sampling_locations = []
sampling_locations_ = []
#把偏移量（x，y）坐标的形式转换成展平的形式，且处理越界问题，当越界时，我们将越界的距离映射到序列的末端
for i in range(sampling_offsets_last1.shape[0]):
    for j in range(4):
        b = np.random.randint(0, 30)
        a = (sampling_offsets_last1[i][j][1]-1) * 19 + sampling_offsets_last1[i][j][0]
        if a > 246.0:
            a = 246.0 - b
        sampling_locations_.append(a)
    sampling_locations.append(sampling_locations_)
    sampling_locations_ = []

#获取最后编码器一层权重的信息
attention_weights_last = attention_weights_list[5]
attention_weights_last1 = attention_weights_last[:, 19700:, :, :]
attention_weights_last1 = attention_weights_last1.to('cpu')
attention_weights_last1 = attention_weights_last1.detach().numpy()
attention_weights_last1 = attention_weights_last1[:, :, 0, 3, :].reshape(247, 4)
attention_weights_all = []
for i in range(247):
    a = attention_weights_last1[i][0] + attention_weights_last1[i][1] + attention_weights_last1[i][2] + attention_weights_last1[i][3]
    attention_weights_all.append(a)

for i in range(247):
    for j in range(4):
        attention_weights_last1[i][j] = attention_weights_last1[i][j] / attention_weights_all[i]
x = np.arange(1, 19*13+1)
y = x[::-1]
x = np.array(([x]))
y = np.array(([y])).T
x_r = np.repeat(x, [19*13], axis=0)
y_r = np.repeat(y, [19*13], axis=1)
z = np.zeros((19*13, 19*13))
for i in range(z.shape[0]):
    for j in range(4):
        z[i][int(sampling_locations[i][j])] = attention_weights_last1[i][j]

x1 = np.arange(1, 170+1)
y1 = x1[::-1]
x1 = np.array(([x1]))
y1 = np.array(([y1])).T
x_r_ = np.repeat(x1, [170], axis=0)
y_r_ = np.repeat(y1, [170], axis=1)
# z_ = np.zeros((170, 170))
# z_[:,:] = z[57:, :170]
z_1 = np.zeros((170, 170))

for i in range(z_1.shape[0]):
    for j in range(z_1.shape[1]):
        z_1[i][j] = z[i][j]

c_ = plt.pcolormesh(x_r_, y_r_, z_1, cmap='viridis_r')# 普通热力图
# c = plt.pcolormesh(x_r, y_r, z, cmap='viridis_r')# 普通热力图
plt.xticks([])    # 去 x 轴刻度
plt.yticks([])    # 去 y 轴刻度
plt.colorbar(c_, label='AUPR')
plt.xlabel('Total sampling points')
plt.ylabel('Total reference points')
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
plt.savefig('/22085400506/DINO-test/img/3.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()



