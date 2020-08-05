import numpy as np
import random
import time
import os
import cv2
import torch
import copy
import json
import argparse
from pathlib import Path
from PIL import Image
from scipy.misc import imread, imsave
from torch.autograd import Variable
from matplotlib import pyplot as plt

from lib.roi_data_layer.minibatch import get_minibatch, get_minibatch
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from lib.model.roi_layers import nms
from lib.model.utils.net_utils import vis_detections
# from roi_data_layer.roidb import combined_roidb

from lib.model.faster_rcnn.fsod import FSOD
# from model.faster_rcnn.qkv import QKVRCNN
# from model.faster_rcnn.faster_rcnn import FasterRCNN

def get_model(net, load_path, n_shot):
    if net == 'fsod':
        model = FSOD(['bg', 'fg'], 50, pretrained=False, num_way=2, num_shot=n_shot)
    # elif net == 'qkv':
    #     model = QKVRCNN(['bg', 'fg'], 50, pretrained=True, num_way=2, num_shot=n_shot, pos_encoding=False)
    # elif args.net == 'fasterrcnn':
    #     imdb, roidb, ratio_list, ratio_index = combined_roidb('voc_2007_trainval')
    #     model = FasterRCNN(imdb.classes, 50, pretrained=True)
    model.create_architecture()
    print("load checkpoint %s" % (load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    model.eval()
    cfg.CUDA = True
    model.cuda()
    
    return model

def prepare_variable():
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    support_ims = torch.FloatTensor(1)

    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    support_ims = support_ims.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    support_ims = Variable(support_ims)

    return im_data, im_info, num_boxes, gt_boxes, support_ims

def support_im_preprocess(im_list, cfg, support_im_size, n_of_shot):
    support_data_all = np.zeros((n_of_shot, 3, support_im_size, support_im_size), dtype=np.float32)
    for i, im in enumerate(im_list):
        im = im[:,:,::-1]  # rgb -> bgr
        target_size = np.min(im.shape[0:2])  # don't change the size
        im, _ = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        _h, _w = im.shape[0], im.shape[1]
        if _h > _w:
            resize_scale = float(support_im_size) / float(_h)
            unfit_size = int(_w * resize_scale)
            im = cv2.resize(im, (unfit_size, support_im_size), interpolation=cv2.INTER_LINEAR)
        else:
            resize_scale = float(support_im_size) / float(_w)
            unfit_size = int(_h * resize_scale)
            im = cv2.resize(im, (support_im_size, unfit_size), interpolation=cv2.INTER_LINEAR)
        h, w = im.shape[0], im.shape[1]
        support_data_all[i, :, :h, :w] = np.transpose(im, (2, 0, 1))
    support_data = torch.from_numpy(support_data_all).unsqueeze(0)
    
    return support_data

def query_im_preprocess(im_data, cfg):
    target_size = cfg.TRAIN.SCALES[0]
    im_data, im_scale = prep_im_for_blob(im_data, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
    im_data = torch.from_numpy(im_data)
    im_info = np.array([[im_data.shape[0], im_data.shape[1], im_scale]], dtype=np.float32)
    im_info = torch.from_numpy(im_info)
    gt_boxes = torch.from_numpy(np.array([0]))
    num_boxes = torch.from_numpy(np.array([0]))
    query = im_data.permute(2, 0, 1).contiguous().unsqueeze(0)
    
    return query, im_info, gt_boxes, num_boxes

def visualize_result(rois, cls_prob, bbox_pred, im_info, im2show, cfg):
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    box_deltas = bbox_pred.data

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4)

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    # re-scale boxes to the origin img scale
    pred_boxes /= im_info[0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    thresh = 0.05
    inds = torch.nonzero(scores[:,1]>thresh).view(-1)
    cls_scores = scores[:,1][inds]
    _, order = torch.sort(cls_scores, 0, True)
    cls_boxes = pred_boxes[inds, :]
    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
    cls_dets = cls_dets[order]
    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
    cls_dets = cls_dets[keep.view(-1).long()]

    im2show = vis_detections(im2show, ' ', cls_dets.cpu().numpy(), 0.5)

    return im2show


if __name__ == '__main__':

    args = parse_args()
    CWD = os.getcwd()

    # support
    support_root_dir = '/home/tony/datasets/supports'
    class_dir = 'dog'
    n_shot = 5
    im_paths = list(Path(os.path.join(support_root_dir, class_dir)).glob('*.jpg'))
    random.seed(0)
    im_path_list = random.sample(im_paths, k=n_shot)
    im_list = []
    fig = plt.figure(num=None, figsize=(8, 8), dpi=50, facecolor='w', edgecolor='k')
    for i, im_path in enumerate(im_path_list):
        im = Image.open(im_path)
        im_list.append(np.asarray(im))   
    support_data = support_im_preprocess(im_list, cfg, 320, n_shot)

    # query
    query_path = '/home/tony/FSOD/data/coco/images/val2014/COCO_val2014_000000416059.jpg'
    im = np.asarray(Image.open(query_path))
    im2show = im.copy()
    query_data, im_info, gt_boxes, num_boxes = query_im_preprocess(im, cfg)

    # prepare data
    data = [query_data, im_info, gt_boxes, num_boxes, support_data]
    im_data, im_info, num_boxes, gt_boxes, support_ims = prepare_variable()
    with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        support_ims.resize_(data[4].size()).copy_(data[4])

    # model
    if args.net == 'fsod':
        model_dir = os.path.join(CWD, 'models/fsod0712_b8_w2_s5/train/checkpoints')
        load_path = os.path.join(model_dir,
                'faster_rcnn_{}_{}_{}.pth'.format(1, 20, 4136))
    elif args.net == 'qkv':
        model_dir = os.path.join(CWD, 'models/qkv0712_b8_w2_s5/train/checkpoints')
        load_path = os.path.join(model_dir,
                'faster_rcnn_{}_{}_{}.pth'.format(1, 28, 4136))
    elif args.net == 'fasterrcnn':
        model_dir = os.path.join(CWD, 'models/fr0712_b8_w2_s5/train/checkpoints')
        load_path = os.path.join(model_dir,
                'faster_rcnn_{}_{}_{}.pth'.format(1, 10, 4136))
    model = get_model(args.net, load_path, n_shot)

    start_time = time.time()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes, support_ims, gt_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    box_deltas = bbox_pred.data

    if args.net == 'fasterrcnn':
        raise Exception('stop inference')

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    # Optionally normalize targets by a precomputed mean and stdev
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4)

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    # re-scale boxes to the origin img scale
    pred_boxes /= data[1][0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    thresh = 0.05
    inds = torch.nonzero(scores[:,1]>thresh).view(-1)
    cls_scores = scores[:,1][inds]
    _, order = torch.sort(cls_scores, 0, True)
    cls_boxes = pred_boxes[inds, :]
    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
    cls_dets = cls_dets[order]
    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
    cls_dets = cls_dets[keep.view(-1).long()]

    end_time = time.time()

    im2show = vis_detections(im2show, ' ', cls_dets.cpu().numpy(), 0.5)
    output_path = os.path.join(CWD, 'output/visualization', 'tmp.jpg')
    cv2.imwrite(output_path, im2show[:, :, ::-1])
    # print(cls_dets.size())
    print(data[1][0][2].item())
    print(cls_dets)
    print(end_time - start_time)
