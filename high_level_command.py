# from single_robotic_arm import SingleRoboticArm
from matplotlib import pyplot as plt
from robotic_arm import SingleRoboticArm
import math
import os
import cv2
import time
import numpy as np
from tqdm import tqdm


shot_pose1 = [250*math.pi/180,-20*math.pi/180,110.01*math.pi/180,-15*math.pi/180,-90*math.pi/180,45*math.pi/180]
shot_pose2 = [250*math.pi/180,-20*math.pi/180,110.01*math.pi/180,-15*math.pi/180,-90*math.pi/180,90*math.pi/180]
shot_pose3 = [250*math.pi/180,-20*math.pi/180,110.01*math.pi/180,-15*math.pi/180,-90*math.pi/180,135*math.pi/180]
shot_pose4 = [250*math.pi/180,-20*math.pi/180,110.01*math.pi/180,-15*math.pi/180,-90*math.pi/180,180*math.pi/180]
shot_pose5 = [250*math.pi/180,-20*math.pi/180,110.01*math.pi/180,-15*math.pi/180,-90*math.pi/180,225*math.pi/180]

def take_rgb(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rob_arm = SingleRoboticArm()
    rgb1 = rob_arm.get_rgb(rob_arm.cam_handle)
    rgb2 = rob_arm.get_rgb(rob_arm.shot_cam_handle)
    cv2.imwrite(os.path.join(save_dir, 'rgb1.jpg'), rgb1)
    cv2.imwrite(os.path.join(save_dir, 'rgb2.jpg'), rgb2)

def take_shot(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    rob_arm = SingleRoboticArm()
    rgb = rob_arm.get_rgb(rob_arm.cam_handle)
    depth = rob_arm.get_depth()
    boxes = rob_arm.get_bbox()
    mask = rob_arm.get_mask(rgb, boxes)
    # plt.imshow(mask[:, :, 0])
    # plt.show()
    grasp_matrices, pcd = rob_arm.get_candidates(mask, depth)
    rob_arm.execute_grasp(grasp_matrices)
    rob_arm.shot_taking(shot_pose1, os.path.join(save_dir, 'shot_1.jpg'))
    rob_arm.shot_taking(shot_pose2, os.path.join(save_dir, 'shot_2.jpg'))
    rob_arm.shot_taking(shot_pose3, os.path.join(save_dir, 'shot_3.jpg'))
    rob_arm.shot_taking(shot_pose4, os.path.join(save_dir, 'shot_4.jpg'))
    rob_arm.shot_taking(shot_pose5, os.path.join(save_dir, 'shot_5.jpg'))
    rob_arm.finish()

def take_particular_query(save_dir, start_n=1000, n_per_cls=1, cls_list=['cube', 'can', 'box', 'bottle']):
    im_dir = os.path.join(save_dir, 'images')
    nd_dir = os.path.join(save_dir, 'ndarray')
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    if not os.path.exists(nd_dir):
        os.makedirs(nd_dir)
    rob_arm = SingleRoboticArm()

    for cls in cls_list:
        for i in tqdm(range(start_n, start_n + n_per_cls)):
            box_list = []
            while len(box_list) == 0:
                rob_arm.settle_particular_obj(setting=cls)
                box_list = rob_arm.get_boxes()
                
            boxes = np.asarray(box_list)
            rgb = rob_arm.get_rgb(rob_arm.cam_handle)
            cv2.imwrite(os.path.join(save_dir, 'images', str(i).zfill(6)+'.jpg'), rgb)
            np.save(os.path.join(save_dir, 'ndarray', str(i).zfill(6)+'.npy'), boxes)
        start_n += n_per_cls
    rob_arm.clear_obj()

def take_query(save_dir, start_n=0, n_per_cls=50):
    im_dir = os.path.join(save_dir, 'images')
    nd_dir = os.path.join(save_dir, 'ndarray')
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    if not os.path.exists(nd_dir):
        os.makedirs(nd_dir)
    rob_arm = SingleRoboticArm()

    cls_list = ['cube', 'can', 'box', 'bottle']
    for cls in cls_list:
        for i in tqdm(range(start_n, start_n + n_per_cls)):
            box_list = []
            while len(box_list) == 0:
                rob_arm.settle_obj(setting=cls)
                box_list = rob_arm.get_boxes()
            
            boxes = np.asarray(box_list)
            rgb = rob_arm.get_rgb(rob_arm.cam_handle)
            cv2.imwrite(os.path.join(save_dir, 'images', str(i).zfill(6)+'.jpg'), rgb)
            np.save(os.path.join(save_dir, 'ndarray', str(i).zfill(6)+'.npy'), boxes)
        start_n += n_per_cls
    rob_arm.clear_obj()

def take_constant(save_dir, start_n=0, n_per_cls=50):
    im_dir = os.path.join(save_dir, 'images')
    nd_dir = os.path.join(save_dir, 'ndarray')
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    if not os.path.exists(nd_dir):
        os.makedirs(nd_dir)
    rob_arm = SingleRoboticArm()

    cls_list = ['cube', 'can', 'box', 'bottle']
    for cls in cls_list:
        for i in tqdm(range(start_n, start_n + n_per_cls)):
            box_list = []
            while len(box_list) == 0:
                rob_arm.settle_constant_obj(setting=cls)
                box_list = rob_arm.get_boxes()
                
            boxes = np.asarray(box_list)
            rgb = rob_arm.get_rgb(rob_arm.cam_handle)
            cv2.imwrite(os.path.join(save_dir, 'images', str(i).zfill(6)+'.jpg'), rgb)
            np.save(os.path.join(save_dir, 'ndarray', str(i).zfill(6)+'.npy'), boxes)
        start_n += n_per_cls
    rob_arm.clear_obj()

def take_query_replace(save_dir, start_n=0, n_per_cls=50, cls_list=['cube', 'can', 'box', 'bottle']):
    im_dir = os.path.join(save_dir, 'images')
    nd_dir = os.path.join(save_dir, 'ndarray')
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    if not os.path.exists(nd_dir):
        os.makedirs(nd_dir)
    rob_arm = SingleRoboticArm()

    for cls in cls_list:
        for i in tqdm(range(start_n, start_n + n_per_cls)):
            box_list = []
            while len(box_list) == 0:
                rob_arm.settle_obj_replace(setting=cls)
                box_list = rob_arm.get_boxes()
            time.sleep(2)
            boxes = np.asarray(box_list)
            rgb = rob_arm.get_rgb(rob_arm.cam_handle)
            cv2.imwrite(os.path.join(save_dir, 'images', str(i).zfill(6)+'.jpg'), rgb)
            np.save(os.path.join(save_dir, 'ndarray', str(i).zfill(6)+'.npy'), boxes)
        start_n += n_per_cls
    rob_arm.clear_obj()

def take_query_dense(save_dir, start_n=0, n_per_cls=50, cls_list=['cube', 'can', 'box', 'bottle']):
    im_dir = os.path.join(save_dir, 'images')
    nd_dir = os.path.join(save_dir, 'ndarray')
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    if not os.path.exists(nd_dir):
        os.makedirs(nd_dir)
    rob_arm = SingleRoboticArm()

    for cls in cls_list:
        for i in tqdm(range(start_n, start_n + n_per_cls)):
            box_list = []
            while len(box_list) == 0:
                rob_arm.settle_obj_dense(setting=cls)
                box_list = rob_arm.get_boxes()
            time.sleep(2)
            boxes = np.asarray(box_list)
            rgb = rob_arm.get_rgb(rob_arm.cam_handle)
            cv2.imwrite(os.path.join(save_dir, 'images', str(i).zfill(6)+'.jpg'), rgb)
            np.save(os.path.join(save_dir, 'ndarray', str(i).zfill(6)+'.npy'), boxes)
        start_n += n_per_cls
    rob_arm.clear_obj()

def take_query_particular_dense(save_dir, start_n=0, n_per_cls=50, cls_list=['cube', 'can', 'box', 'bottle']):
    im_dir = os.path.join(save_dir, 'images')
    nd_dir = os.path.join(save_dir, 'ndarray')
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    if not os.path.exists(nd_dir):
        os.makedirs(nd_dir)
    rob_arm = SingleRoboticArm()

    for cls in cls_list:
        for i in tqdm(range(start_n, start_n + n_per_cls)):
            rob_arm.settle_obj_particular_dense(setting=cls)
            time.sleep(2)
            rgb = rob_arm.get_rgb(rob_arm.cam_handle)
            cv2.imwrite(os.path.join(save_dir, 'images', str(i).zfill(6)+'.jpg'), rgb)
        start_n += n_per_cls
    rob_arm.clear_obj()

def get_object_size():
    rob_arm = SingleRoboticArm()
    cls_list = ['cube', 'can', 'box', 'bottle']
    for cls in cls_list:
        for i in range(2):
            obj = rob_arm.place_obj(cls)
            rob_arm.obj_list.append((obj, cls))
    box_list = rob_arm.get_boxes()
    rob_arm.clear_obj()


def main():
    rob_arm = SingleRoboticArm()
    home_pose = [-165*math.pi/180,18*math.pi/180,70*math.pi/180,20*math.pi/180,-90*math.pi/180,90*math.pi/180]
    rob_arm.MoveJointPosition(home_pose)
    time.sleep(1)

    rgb = rob_arm.get_rgb(rob_arm.cam_handle)
    depth = rob_arm.get_depth()
    boxes = rob_arm.get_bbox()
    mask = rob_arm.get_mask(rgb, boxes)
    grasp_matrices, pcd = rob_arm.get_candidates(mask, depth)
    rob_arm.execute_grasp(grasp_matrices)

    rob_arm.ReleaseGripper()
    time.sleep(2)
    
    # up_pose = [-140*math.pi/180,18*math.pi/180,58*math.pi/180,30*math.pi/180,-90*math.pi/180,105*math.pi/180]
    # rob_arm.MoveJointPosition(up_pose)

    # shot_pose1 = [-115*math.pi/180,9*math.pi/180,70*math.pi/180,18*math.pi/180,-90*math.pi/180,45*math.pi/180]
    # shot_pose3 = [-115*math.pi/180,9*math.pi/180,70*math.pi/180,18*math.pi/180,-90*math.pi/180,105*math.pi/180]
    # shot_pose5 = [-115*math.pi/180,9*math.pi/180,70*math.pi/180,18*math.pi/180,-90*math.pi/180,165*math.pi/180]
    # rob_arm.MoveJointPosition(shot_pose1)
    # rob_arm.MoveJointPosition(shot_pose3)
    # rob_arm.MoveJointPosition(shot_pose5)

    # place_pose = [-190*math.pi/180,18*math.pi/180,65*math.pi/180,20*math.pi/180,-90*math.pi/180,90*math.pi/180]
    # rob_arm.MoveJointPosition(place_pose)

    # place_pose = [-190*math.pi/180,18*math.pi/180,58*math.pi/180,30*math.pi/180,-90*math.pi/180,90*math.pi/180]
    # rob_arm.MoveJointPosition(place_pose)

    # rob_arm.shot_taking(shot_pose1, os.path.join(save_dir, 'shot_1.jpg'))
    # rob_arm.shot_taking(shot_pose2, os.path.join(save_dir, 'shot_2.jpg'))
    # rob_arm.shot_taking(shot_pose3, os.path.join(save_dir, 'shot_3.jpg'))
    # rob_arm.shot_taking(shot_pose4, os.path.join(save_dir, 'shot_4.jpg'))
    # rob_arm.shot_taking(shot_pose5, os.path.join(save_dir, 'shot_5.jpg'))
    rob_arm.finish()


if __name__ == '__main__':
    # get_object_size()

    # save_dir = '/home/tony/YCB_simulation/support'
    # take_shot(os.path.join(save_dir, 'ball'))

    # save_dir = '/home/tony/YCB_simulation/query'
    # take_particular_query(save_dir, start_n=1000, n_per_cls=256, cls_list=['cube', 'can', 'box', 'bottle'])

    # save_dir = '/home/tony/YCB_simulation/query'
    # take_constant(save_dir, start_n=100000, n_per_cls=64)

    # save_dir = '/home/tony/YCB_simulation/query'
    # take_query_replace(save_dir, start_n=101024, n_per_cls=256, cls_list=['cube', 'can', 'box', 'bottle'])
    # save_dir = '/home/tony/YCB_simulation/query'
    # take_query_dense(save_dir, start_n=225, n_per_cls=25, cls_list=['cube'])
    # save_dir = '/home/tony/YCB_simulation/query'
    # take_query_particular_dense(save_dir, start_n=5512, n_per_cls=256, cls_list=['box'])

    # take_rgb('/home/tony/Desktop/vrep_image')


    main()