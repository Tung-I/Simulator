try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import numpy as np
import cv2
import math
import time
import random #random.shuffle
import os 
import os.path as osp
from grasp_candidate import Grasper


g = Grasper()
det_boxes = []
det_angles = []

focal_length = 229.4516463256709
principal = 128
intrinsic = [[focal_length, 0, principal],[0, focal_length, principal],[0, 0, 1]]
extrinsic = None

def calExtrinsic(cam_pos, cam_quat):
    qx, qy, qz, qw = cam_quat
    tx, ty, tz = cam_pos
    extrinsic = [
        [1-2*(qy**2)-2*(qz**2), 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw, tx],
        [2*qx*qy+2*qz*qw, 1-2*(qx**2)-2*(qz**2), 2*qy*qz-2*qx*qw, ty],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*(qx**2)-2*(qy**2), tz]
    ]
    return extrinsic


def graspGDN(mask, depth_img, cam_pos, cam_quat, cam_mat):
    extrinsic = calExtrinsic(cam_pos, cam_quat)
    mask = np.any(mask, axis=2)

    pcd = g.build_point_cloud(depth_img, intrinsic, extrinsic)
    grasping_candidates = g.get_grasping_candidates(pcd, mask)
    
    return grasping_candidates, pcd


def setObjPosQat(clientID, obj_name, target_pos, target_qat):

    _, obj_handle = vrep.simxGetObjectHandle(clientID, obj_name, vrep.simx_opmode_oneshot_wait)

    vrep.simxSetObjectPosition(clientID, obj_handle, -1, target_pos, vrep.simx_opmode_oneshot)
    vrep.simxSetObjectQuaternion(clientID, obj_handle, -1, target_qat, vrep.simx_opmode_oneshot)
    time.sleep(0.5)


def getObjPosQat(clientID, obj_name):

    _, obj_handle = vrep.simxGetObjectHandle(clientID, obj_name, vrep.simx_opmode_oneshot_wait)

    _, obj_pos = vrep.simxGetObjectPosition(clientID, obj_handle, -1, vrep.simx_opmode_oneshot_wait)
    _, obj_qat = vrep.simxGetObjectQuaternion(clientID, obj_handle, -1, vrep.simx_opmode_oneshot_wait)
    
    return obj_handle, obj_pos, obj_qat


# random put objects in the scene from camera pos
def randomDropObj(clientID, cam_pos):
    dropping_pos = cam_pos.copy()
    dropping_pos[1] -= 0.2

    for obj_filename in [i for i in os.listdir('YCB') if '.ttm' in i][:10]:
        _, obj = vrep.simxLoadModel(clientID, './YCB/%s' % obj_filename, 0, vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(clientID, obj, -1, dropping_pos, vrep.simx_opmode_blocking)

    time.sleep(3)


# rgb image of whole scene
def getRGB(clientID, cam_handle):

    _, resolution, raw_image = vrep.simxGetVisionSensorImage(clientID, cam_handle, 0, vrep.simx_opmode_blocking)
    
    color_img = np.asarray(raw_image)
    color_img.shape = (resolution[1], resolution[0], 3)
    color_img = color_img.astype(np.float)
    color_img[color_img < 0] += 255
    color_img = cv2.flip(color_img,0).astype(np.uint8)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    
    return color_img


# depth image of whole scene
def getDepth(clientID, cam_handle):

    _, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_handle, vrep.simx_opmode_blocking)
    
    depth_img = np.asarray(depth_buffer)
    depth_img.shape = (resolution[1], resolution[0])
    depth_img = cv2.flip(depth_img,0).astype(np.float)
    Near_plane = 0.01
    Far_plane = 0.5
    depth_img = (Far_plane - Near_plane) * depth_img + Near_plane
        
    depth_img_show = depth_img - np.min(depth_img)
    depth_img_show /= np.max(depth_img_show)
    depth_img_show = (depth_img_show*255.0).astype(np.uint8)

    # cv2.imwrite('depth_img.png', depth_img_show)
    
    return depth_img


    img = input_img.copy()

    img = img - np.min(img)
    img /= np.max(img)
    img = (img*255.0).astype(np.uint8)
    
    return img