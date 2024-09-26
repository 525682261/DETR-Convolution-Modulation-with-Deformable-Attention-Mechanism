import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
model_checkpoint_path = "/22085400506/DyConv1d-dis-lf2/logs/DINO/R50-MS4-dis-Dyconv1d-lf2/checkpoint0011.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# load coco names
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}


args.dataset_file = 'coco'
args.coco_path = "/22085400506/dataset/coco" # the path of coco
args.fix_size = False

dataset_val = build_dataset(image_set='val', args=args)

image, targets = dataset_val[122]

# figure_save_path = "/22085400506/DINO/logs/DINO"
# plt.savefig(os.path.join(figure_save_path, image))#第一个是指存储路径，第二个是图片名字
# plt.show()

# build gt_dict for vis
box_label = [id2name[int(item)] for item in targets['labels']]
gt_dict = {
    'boxes': targets['boxes'],
    'image_id': targets['image_id'],
    'size': targets['size'],
    'box_label': box_label,
}
vslzr = COCOVisualizer()
vslzr.visualize(image, gt_dict, dpi=1200)
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

thershold = 0.4 # set a thershold

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': targets['size'],
    'box_label': box_label,
    'image_id':gt_dict['image_id']
}
vslzr.visualize(image, pred_dict, savedir='/22085400506/our-f',dpi=1200,show_in_console=False)


# from PIL import Image
# import datasets.transforms as T
#
# image = Image.open("./figs/1cdf.jpg").convert("RGB") # load image
#
# # transform images
# transform = T.Compose([
#     T.RandomResize([800], max_size=1333),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# image, _ = transform(image, None)
#
# # predict images
# output = model.cuda()(image[None].cuda())
# output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
#
# # visualize outputs
# thershold = 0.065 # set a thershold
#
# vslzr = COCOVisualizer()
#
# scores = output['scores']
# labels = output['labels']
# boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
# select_mask = scores > thershold
#
# box_label = [id2name[int(item)] for item in labels[select_mask]]
# pred_dict = {
#     'boxes': boxes[select_mask],
#     'size': torch.Tensor([image.shape[1], image.shape[2]]),
#     'box_label': box_label
# }
# vslzr.visualize(image, pred_dict, savedir=None, dpi=1200)


