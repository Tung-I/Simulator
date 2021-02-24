import sys
import numpy as np
import math

'''
For GDN
'''
import json
from gdn.representation.euler import *
from gdn.detector.edgeconv.backbone import EdgeDet
from nms import decode_euler_feature
from nms import initEigen, sanity_check
from nms import crop_index, generate_gripper_edge
from scipy.spatial.transform import Rotation
import cv2

import time
from datetime import datetime
initEigen(0)


class Grasper(object):
    def __init__(self, gdn_config_path = "/home/tony/VREP-env/edgeconv.json", gdn_weight_path = "/home/tony/VREP-env/edgeconv.ckpt"):
        with open(gdn_config_path, "r") as fp:
            self.gdn_config = json.load(fp)
            self.gdn_config['thickness'] = 0.03 # Force overwrite
        self.gdn_gripper_length = self.gdn_config['hand_height']
        self.gdn_input_points = self.gdn_config['input_points']
        self.gdn_model = EdgeDet(self.gdn_config, activation_layer=EulerActivation())
        self.gdn_model = self.gdn_model.cuda()
        self.gdn_model = self.gdn_model.eval()
        self.gdn_model.load_state_dict(torch.load(gdn_weight_path)['base_model'])
        self.gdn_representation = EulerRepresentation(self.gdn_config)
        self.gdn_subsampling_util = val_collate_fn_setup(self.gdn_config)

    def get_grasping_candidates(self, point_cloud, segmentation):
        '''
        point_cloud: Point cloud in world coordinate. Shape: (im_height, im_width, 3), float32
        segmentation: Mask of the object to grasp. Shape: (im_height, im_width), boolean
        '''
        grasping_candidates = []

        # point_cloud = build_point_cloud(depth_img, intrinsic, trans_mat).astype(np.float32)
        # get the segmentation image
        # seg_img = self._camera.get_image(sensor_type=['seg'])['seg'] #(H, W, 3)
        # segmentation = np.any(seg_img, axis=2) # (H, W)
        # get partial point cloud
        pc_npy = point_cloud[segmentation,:]
        pc_npy_max = np.max(pc_npy, axis=0)
        pc_npy_min = np.min(pc_npy, axis=0) 
        trans_to_frame = (pc_npy_max + pc_npy_min) / 2.0
        trans_to_frame[2] = np.min(pc_npy[:,2])
        objs_value_range = (pc_npy_max - pc_npy_min).max()

        while pc_npy.shape[0] < self.gdn_input_points:
            new_pts = pc_npy[np.random.choice(len(pc_npy), self.gdn_input_points-len(pc_npy), replace=True)]
            new_pts = new_pts + np.random.randn(*new_pts.shape) * 1e-6
            pc_npy = np.append(pc_npy, new_pts, axis=0)
            pc_npy = np.unique(pc_npy, axis=0)
        if pc_npy.shape[0]>self.gdn_input_points:
            pc_npy = pc_npy[np.random.choice(len(pc_npy), self.gdn_input_points, replace=False),:]

        # generate grasping candidates
        pc_npy -= trans_to_frame
        pc_batch, indices, reverse_lookup_index, _ = self.gdn_subsampling_util([(pc_npy,None),])
        with torch.no_grad():
            pred = self.gdn_model(pc_batch.cuda(), [pt_idx.cuda() for pt_idx in indices]).cpu().numpy()
        grasping_candidates = np.asarray(decode_euler_feature(
          pc_npy[reverse_lookup_index[0]]+trans_to_frame[np.newaxis],
          pred[0].reshape(1,-1),
          pred.shape[1],pred.shape[2],pred.shape[3],
          self.gdn_config['hand_height'],
          self.gdn_config['gripper_width'],
          self.gdn_config['thickness_side'],
          self.gdn_config['rot_th'],
          self.gdn_config['trans_th'],
          10000, # max number of candidate
          -np.inf, # threshold of candidate
          5000,  # max number of grasp in NMS
          20,    # number of threads
          True  # use NMS
        ), dtype=np.float32)

        grasping_candidates = sanity_check(point_cloud.reshape(-1, 3), grasping_candidates, 10,
          self.gdn_config['gripper_width'],
          self.gdn_config['thickness'],
          self.gdn_config['hand_height'],
          self.gdn_config['thickness_side'],
          12 # num threads
        )
        
        return grasping_candidates

    def build_point_cloud(self, depth_img, intrinsic, trans_mat):
        inv_intrinsic = np.linalg.pinv(intrinsic) # (3, 3)
        y = np.arange(depth_img.shape[0]-1, -0.5, -1)
        x = np.arange(depth_img.shape[1]-1, -0.5, -1)
        xv, yv = np.meshgrid(x, y)
        xy = np.append(xv[np.newaxis], yv[np.newaxis], axis=0) # (2, H, W)
        xy_homogeneous = np.pad(xy, ((0,1),(0,0),(0,0)), mode='constant', constant_values=1) # (3, H, W)
        xy_homogeneous_shape = xy_homogeneous.shape
        xy_h_flat = xy_homogeneous.reshape(3, -1) # (3, H*W) 
        xy_h_flat_t = np.dot(inv_intrinsic, xy_h_flat) # (3,3) x (3, H*W) -> (3, H*W)
        xy_homogeneous_t = xy_h_flat_t.reshape(xy_homogeneous_shape) # (3, H, W)
        
        xyz_T = (xy_homogeneous_t * depth_img[np.newaxis]).reshape(3, -1) # (3, H*W)
        xyz_T_h = np.pad(xyz_T, ((0,1), (0,0)), mode='constant', constant_values=1) # (4, H*W)
        xyz = np.dot(trans_mat, xyz_T_h).reshape(xy_homogeneous_shape) # (3, H, W)
        xyz = np.transpose(xyz, (1, 2, 0)) # (H, W, 3)
        return xyz

if __name__ == '__main__':
    

    g = Grasper()
    
    ### test
    # pcd = np.random.randn(100,100,3).astype(np.float32)
    # mask = np.random.randn(100,100).astype(np.bool)

    intrinsic = [[229.4516463256709, 0, 128], [0, 229.4516463256709, 128], [0, 0, 1]]
    extrinsic = [[1.0, 0.0, 0.0, -0.12500008940696716], 
                 [0.0, -0.9659259228832937, -0.2588189190257495, -0.3749999403953552], 
                 [0.0, 0.2588189190257495, -0.9659259228832937, 0.42499998211860657]]
    
    # extrinsic = [1.0, 0.0, 0.0, -0.12500008940696716, 0.0, -0.965925931930542, -0.25881892442703247, -0.3749999403953552, 0.0, 0.25881892442703247, -0.965925931930542, 0.42499998211860657]
    
    depth_img = cv2.imread('depth_img.png', 0)
    depth_img = depth_img.astype(np.float32)
    depth_img = depth_img/255
    

    mask_img = cv2.imread('mask_img.png', 0)
    
    mask = np.zeros((256, 256), dtype=np.bool)
    mask[mask_img > 0] = True
    
    print(mask.shape, np.unique(mask))
    pcd = g.build_point_cloud(depth_img, intrinsic, extrinsic).astype(np.float32)
    np.save('dump_all.npy', pcd.reshape(-1, 3))
    
    print(g.get_grasping_candidates(pcd, mask))
