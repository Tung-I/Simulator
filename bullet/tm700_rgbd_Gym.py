import random
import os
from gym import spaces
import time
import json
import pybullet as p
import numpy as np
import pybullet_data
import pdb
import torch
import distutils.dir_util
import glob
from pathlib import Path
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from bullet.tm700_possensor_Gym import tm700_possensor_gym
from matplotlib import pylab as plt
import cv2

from lib.model.utils.config import cfg
from detector import get_model, prepare_variable, support_im_preprocess, query_im_preprocess, visualize_result

cwd = os.getcwd()
DATA_ROOT = os.path.join(cwd, 'ShapeNet_3obj')
random.seed()

class tm700_rgbd_gym(tm700_possensor_gym):
    """Class for tm700 environment with diverse objects.
    """
    def __init__(self,
                urdfRoot=pybullet_data.getDataPath(),
                objRoot='',
                actionRepeat=80,
                isEnableSelfCollision=True,
                renders=False,
                isDiscrete=True,
                maxSteps=11,
                dv=0.06,
                removeHeightHack=False,
                blockRandom=0.30,
                cameraRandom=0,
                width=64,
                height=64,
                numObjects=1,
                isTest=False):
        """Initializes the tm700DiverseObjectEnv.

        Args:
        urdfRoot: The diretory from which to load environment URDF's.
        actionRepeat: The number of simulation steps to apply for each action.
        isEnableSelfCollision: If true, enable self-collision.
        renders: If true, render the bullet GUI.
        isDiscrete: If true, the action space is discrete. If False, the
            action space is continuous.
        maxSteps: The maximum number of actions per episode.
        dv: The velocity along each dimension for each action.
        removeHeightHack: If false, there is a "height hack" where the gripper
            automatically moves down for each action. If true, the environment is
            harder and the policy chooses the height displacement.
        blockRandom: A float between 0 and 1 indicated block randomness. 0 is
            deterministic.
        cameraRandom: A float between 0 and 1 indicating camera placement
            randomness. 0 is deterministic.
        width: The image width.
        height: The observation image height.
        numObjects: The number of objects in the bin.
        isTest: If true, use the test set of objects. If false, use the train
            set of objects.
        """

        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._objRoot = objRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._dv = dv
        self._p = p
        self._removeHeightHack = removeHeightHack
        self._blockRandom = blockRandom
        self._cameraRandom = cameraRandom
        self._width = width
        self._height = height
        self._numObjects = numObjects
        self._isTest = isTest
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self._height, self._width, 3),
                                            dtype=np.uint8)

        # get .obj files paths and their classes
        self._objList, self._objNameList = self._get_all_obj()
        self.get_shot_random_seed = 6

        # detection related
        self.support_data_pool = {}
        self.n_shot = 0
        self.model_dir = '/home/tony/FSOD/models/simulated/train/checkpoints'
        self.model_load_path = os.path.join(self.model_dir,
                'faster_rcnn_{}_{}_{}.pth'.format(1, 28, 1248))
        self.model_name = 'fsod'
        self.detection_model = get_model(self.model_name, self.model_load_path, self.n_shot)

        # rendering
        if self._renders:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33]) # cameraposition of rendering
        else:
            self.cid = p.connect(p.DIRECT)
        self.seed()

        if (self._isDiscrete):
            if self._removeHeightHack:
                self.action_space = spaces.Discrete(9)
            else:
                self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
            if self._removeHeightHack:
                self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
        self.viewer = None

    def _get_all_obj(self):
        """Return file paths.
        """
        root = Path(DATA_ROOT)
        class_dirs = [c for c in root.iterdir() if c.is_dir() and 'ipynb' not in str(c)]
        obj_list = []
        obj_name_list = []
        for c_d in class_dirs:
            obj_name_list.append(c_d.parts[-1])
            obj_dirs = [sub_dir for sub_dir in c_d.iterdir() if sub_dir.is_dir()]
            random.seed(1024)
            _dir = random.sample(obj_dirs, k=1)[0]
            obj_path = str(_dir / Path('models/model_normalized.obj'))
            obj_list.append(obj_path)
        
        return obj_list, obj_name_list

    def reset(self):
        """Environment reset called at the beginning of an episode.
        """
        # Set the camera settings.
        look = [0.00, -0.15, 0.60]
        self._cam_pos = look
        distance = 0.1
        pitch = -45
        yaw = -75
        roll = 120
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        self.fov = 35.
        self.focal_length_x = self._width / np.tan(np.radians(self.fov)/2.0)
        self.focal_length_y = self._height / np.tan(np.radians(self.fov)/2.0)
        aspect = self._width / self._height
        self.d_near = 0.01
        self.d_far = 1.5
        self._proj_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.d_near, self.d_far)
        self._attempted_grasp = False
        self._env_step = 0
        self.terminated = 0

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        self.table_pose = [.5000000, 0.00000, -.640000, 0.000000, 0.000000, 0.0, 1.0]
        self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), *self.table_pose)
        p.setGravity(0, 0, -10)
        self._tm700 = tm700(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._objectUids = self._randomly_place_objects(self._objList)

        # detection
        self.n_shot += 1
        self._observation, query_rgb = self._get_observation()
        target_cls_name = 'mug'
        support_im_dict = self._get_all_obj_shot(self._objList)
        for k in support_im_dict.keys():
            support_im_list = support_im_dict[k]
            if k in self.support_data_pool.keys():
                self.support_data_pool[k] += support_im_list
            else:
                self.support_data_pool[k] = support_im_list
        support_im_list = self.support_data_pool[target_cls_name]
        support_data = support_im_preprocess(support_im_list, cfg, 320, self.n_shot)
        query_data, im_info, gt_boxes, num_boxes = query_im_preprocess(query_rgb, cfg)
        data = [query_data, im_info, gt_boxes, num_boxes, support_data]
        im_data, im_info, num_boxes, gt_boxes, support_ims = prepare_variable()
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])
            support_ims.resize_(data[4].size()).copy_(data[4])
    
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.detection_model(im_data, im_info, gt_boxes, num_boxes, support_ims, gt_boxes, n_shot=self.n_shot)
        im2show = query_rgb.copy()
        im_result, dets = visualize_result(rois, cls_prob, bbox_pred, im_info, im2show, cfg)

        # im = support_data[0][:, :, ::-1]/255
        print(f'current number of shots: {support_ims.size(1)}')
        print(dets)
        cv2.imshow('support', self.support_data_pool[target_cls_name][self.n_shot - 1][:, :, ::-1] / 255.)
        cv2.imshow('result', im_result[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return np.array(self._observation)
  

    def _get_all_obj_shot(self,objList):
        """Take shots of all the objects
        """
        # create a table for taking a shot of object
        shot_table_pose = [.5000000, 1.50000, -.640000, 0.000000, 0.000000, 0.0, 1.0]
        shotTableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), *shot_table_pose)
        p.changeVisualShape(objectUniqueId=shotTableUid,linkIndex=-1,rgbaColor=[0,0,0,0])
        # rgb ims to return
        num_of_obj = 3
        shot_per_obj = 1
        support_im_size = 320
        support_im_dict = {}  # {cls_name: a list of im array}
        # position
        shift = [0, -0.02, 0]
        meshScale = [0.2, 0.2, 0.2]
        x_pos = 0.5
        y_pos = 1.5
        for idx, obj_path in enumerate(objList):
            current_obj_cls_name = str(Path(obj_path).parts[-4])
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=obj_path,
                                                rgbaColor=[1, 1, 1, 1],
                                                specularColor=[0.4, .4, 0],
                                                visualFramePosition=shift,
                                                meshScale=meshScale)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=obj_path,
                                                collisionFramePosition=shift,
                                                meshScale=meshScale)
            uid = p.createMultiBody(baseMass=1,
                                baseInertialFramePosition=[-0.2, 0, 0],
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=[x_pos, y_pos, 0.15],
                                useMaximalCoordinates=True)

            for _ in range(100):
                p.stepSimulation()
            yaw = 0
            # one obj, one shot
            shot_list = []
            for i in range(shot_per_obj):
                look = [0.5, 1.5, 0]
                distance = 0.65
                self.get_shot_random_seed += 1
                random.seed(self.get_shot_random_seed)
                pitch = random.randint(40,60)*-1
                yaw = random.randint(0,360)*-1
                roll = 0
                # if i < 6:
                #     look = [0.5, 1.5, 0]
                #     distance = 0.65
                #     pitch = 0
                #     roll = 0
                # elif i >=6 and i < 12:
                #     look = [0.5, 1.5, 0.35]
                #     distance = 0.35
                #     pitch = -90
                #     roll = 0
                # else:
                #     look = [0.5, 1.5, 0]
                #     distance = 0.65
                #     pitch = -50
                #     roll = 0
                view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
                # yaw -= 60
                img_arr = p.getCameraImage(support_im_size, support_im_size, view_matrix, self._proj_matrix)
                w = img_arr[0]  # width of the image, in pixels
                h = img_arr[1]  # height of the image, in pixels
                rgb = img_arr[2]  # color data RGB
                np_im_arr = np.reshape(rgb, (h, w, 4))
                np_im_arr = np_im_arr[:,:,:3]
                # img = cv2.cvtColor(np_img_arr,cv2.COLOR_RGB2BGR)
                # cv2.imwrite('shot/obj'+str(idx+1)+'-'+str(i+1)+'.png', img)
                x_min, y_min, x_max, y_max = self.clip_sample(np_im_arr)
                np_im_arr = np_im_arr[ y_min:y_max+1, x_min:x_max+1, :]
                np_im_arr[np.where(np_im_arr==255.)] = 0
                shot_list.append(np_im_arr)
            p.removeBody(uid)
            support_im_dict[current_obj_cls_name] = shot_list
        p.removeBody(shotTableUid)

        return support_im_dict


    def _randomly_place_objects(self, objList):
        """Randomly places the objects in the bin.
        """
        objectUids = []
        shift = [0, -0.02, 0]
        meshScale = [0.2, 0.2, 0.2]
        list_x_pos = [0.55, 0.6, 0.65]
        list_y_pos = [0.2, 0.05, 0.2]
        # list_x_pos = [random.randint(45,70)/100 for i in range(3)]
        # list_y_pos = [random.randint(0,20)/100 for i in range(3)]
        for idx, obj_path in enumerate(objList):
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                                fileName=obj_path,
                                                rgbaColor=[1, 1, 1, 1],
                                                specularColor=[0.4, .4, 0],
                                                visualFramePosition=shift,
                                                meshScale=meshScale)
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=obj_path,
                                                collisionFramePosition=shift,
                                                meshScale=meshScale)
            xpos = list_x_pos[idx]
            ypos = list_y_pos[idx]
            uid = p.createMultiBody(baseMass=1,
                            baseInertialFramePosition=[-0.2, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            basePosition=[xpos, ypos, 0.15],
                            useMaximalCoordinates=True)
            objectUids.append(uid)

            for _ in range(500):
                p.stepSimulation()

        return objectUids


    def _get_observation(self):
        """Return the observation as an image.
        """
        img_arr = p.getCameraImage(width=self._width,
                                height=self._height,
                                viewMatrix=self._view_matrix,
                                projectionMatrix=self._proj_matrix)
        rgb = img_arr[2]
        depth = img_arr[3]
        min = 0.97
        max=1.0
        depthnormalized = [(i - min)/(max-min) for i in depth]
        segmentation = img_arr[4]
        depth = np.reshape(depthnormalized, (self._height, self._width,1) )
        segmentation = np.reshape(segmentation, (self._height, self._width,1) )
        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        np_img_arr = np_img_arr.astype(np.float64)
        test = np.concatenate([np_img_arr[:, :, 1:3], depth], axis=-1)

        return test, rgb[:, :, 0:3]


    def clip_sample(self, support_im):
        for x in range(320):
            col = support_im[:, x, :]
            if col.sum() != 255. * 320 * 3:
                x_max = x
        for x in range(319, 0, -1):
            col = support_im[:, x, :]
            if col.sum() != 255. * 320 * 3:
                x_min = x
        for y in range(320):
            row = support_im[y, :, :]
            if row.sum() != 255. * 320 * 3:
                y_max = y
        for y in range(319, 0, -1):
            row = support_im[y, :, :]
            if row.sum() != 255. * 320 * 3:
                y_min = y
        return [x_min, y_min, x_max, y_max]

  
