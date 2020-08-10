import json
import os
import re
import fnmatch
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm


# transform np array into jpg
np_im_dir = Path('/home/tony/datasets/simulated_data/np_img')
for _n in tqdm(range(1, 5000+1)):
    n_6d = "%06d" % _n
    np_im_path = list(np_im_dir.glob('*_'+str(_n-1)+'.npy'))[0]
    im = np.load(np_im_path)
    im_save_dir = Path('/home/tony/datasets/simulated_data/images/simulated2020')
    cv2.imwrite(str(im_save_dir/ Path(n_6d+'.jpg')), im[:, :, ::-1])

coco_json_path = '/home/tony/datasets/coco/annotations/instances_minival2014.json'
with open(coco_json_path, 'r') as f:
    data = json.load(f)
data.keys()

NAME2IDX = {
    'airplane': 1, 'car': 2, 'guitar': 3, 'laptop': 4, 'pistol': 5, 'bag': 6, 'chair': 7, 'knife': 8, 'motorbike': 9, 'rocket': 10, 
    'table': 11, 'cap': 12, 'earphone': 13, 'lamp': 14, 'mug': 15, 'skateboard': 16}
IDX2NAME = [
    'background', 'airplane', 'car', 'guitar', 'laptop', 'pistol', 'bag', 'chair', 'knife', 'motorbike', 'rocket', 
    'table', 'cap', 'earphone', 'lamp', 'mug', 'skateboard']

# create category
new_data_categories = []
for i in tqdm(range(16)):   
    cat_id  = i + 1
    dic = {}
    dic['supercategory'] = 'DIY'
    dic['id'] = cat_id
    dic['name'] = IDX2NAME[cat_id]
    new_data_categories.append(dic)

# create images
new_data_images = []
for i in tqdm(range(1, 5000+1)):
    n_6d = "%06d" % i
    dic = {}
    dic['license'] = 1
    dic['file_name'] = n_6d + '.jpg'
    dic['coco_url'] = 'http://farm3.staticflickr.com/2253/1755223462_fabbeb8dc3_z.jpg'
#     im = cv2.imread('/home/tony/datasets/simulated_data/images/' + dic['file_name'])
#     dic['height'] = im.shape[0]
#     dic['width'] = im.shape[1]
    dic['height'] = 320
    dic['width'] = 320
    dic['date_captured'] = '2013-11-15 13:55:22'
    dic['flickr_url'] = 'http://farm8.staticflickr.com/7007/6413705793_1c391cd697_z.jpg'
    dic['id'] = i  # id of 1.jpg is 1
    new_data_images.append(dic)

# create annotations
new_data_annotations = []
cnt = 0
for i in tqdm(range(1, 5000+1)):
    img_id = i
    img_path_6d = "%06d" % i
    dets = np.load('/home/tony/datasets/simulated_data/np_anno/annotation_' + str(i-1) + '.npy')
    for j in range(dets.shape[0]):
        box_info = dets[j]
        x, y = box_info[0], box_info[1]
        w, h = box_info[2] - x, box_info[3] - y
        dic = {}
        dic['segmentation'] = [[184.05]]
        dic['area'] = 1.28
        dic['iscrowd'] = 0
        dic['image_id'] = img_id
        dic['bbox'] = [x, y, w, h]
        dic['category_id'] = int(box_info[4])
        cnt += 1
        dic['id'] = cnt 
        new_data_annotations.append(dic)

coco_json_path = '/home/tony/datasets/coco/annotations/instances_train2014.json'
with open(coco_json_path, 'r') as f:
    data = json.load(f)
new_data_info = data['info']
new_data_licenses = data['licenses']

new_dict = {}
new_dict['info'] = new_data_info
new_dict['images'] = new_data_images
new_dict['licenses'] = new_data_licenses
new_dict['annotations'] = new_data_annotations
new_dict['categories'] = new_data_categories

dump_path = '/home/tony/datasets/simulated_data/annotations/train.json'
with open(dump_path, 'w') as f:
    json.dump(new_dict, f)

from pycocotools.coco import COCO
_COCO = COCO(dump_path)

print(len(_COCO.imgs))
print(len(_COCO.anns))
print(len(_COCO.cats))