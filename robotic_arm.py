# simExtRemoteApiStart(19999)
try:
    # import sim as vrep
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
import os
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from grasp_candidate import Grasper
g = Grasper()

# from gdn_grasp.grasp_candidate import Grasper


class SingleRoboticArm():
    def __init__(self):
        print ('Program started')
        vrep.simxFinish(-1) # just in case, close all opened connections

        while True:
            self.clientID = vrep.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to V-REP
            if self.clientID != -1:
                print ('Connected to remote API server')
                returnCode = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_oneshot)
                break
            else:
                print ('Failed connecting to remote API server')


        time.sleep(1)
        self.emptyBuff = bytearray()

        sim_ret, self.robot_handle = vrep.simxGetObjectHandle(self.clientID, "UR5", vrep.simx_opmode_oneshot_wait)

        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor_0", vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_handle_1 = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor_1", vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_handle_2 = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor_2", vrep.simx_opmode_oneshot_wait)


        sim_ret, self.cam_pos = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        
        

        self.original_matrix = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],np.float)
        self.original_matrix_gdn = np.array([[-1,0,0,0],
                                             [0,-1,0,0],
                                             [0,0,1,0],
                                             [0,0,0,1]],np.float) 
        self.focal_length = 309.019
        self.principal = 128

        # self.obj_filenames =  [i for i in os.listdir('YCB') if '.ttm' in i]
        self.obj_list = []
        self.cls2label = {'cube':1, 'can':2, 'box':3, 'bottle':4}
    
    def MoveJointPosition(self, send_matrix):
        res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'SendJointPosition',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
        running = True
        while running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_JointPositionMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False
    def ReleaseGripper(self):
        res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'SendReleaseSignal',[],[1],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
        running = True
        while running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_ReleaseSignal'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False

    def shot_taking(self, action_matrix, save_path):
        send_matrix = action_matrix
        res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'ShotMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
        running = True
        while running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_ShotMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False
        rgb = self.get_rgb(self.shot_cam_handle)
        cv2.imwrite(save_path, rgb)

    def get_candidates(self, mask, depth_img):
        def calExtrinsic(cam_pos, cam_quat):
            qx, qy, qz, qw = cam_quat
            tx, ty, tz = cam_pos
            extrinsic = [
                [1-2*(qy**2)-2*(qz**2), 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw, tx],
                [2*qx*qy+2*qz*qw, 1-2*(qx**2)-2*(qz**2), 2*qy*qz-2*qx*qw, ty],
                [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*(qx**2)-2*(qy**2), tz]
            ]
            return extrinsic
        res, retInts, cam_matrix, retStrings, retBuffer = \
            vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[self.cam_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        sim_ret, cam_pos = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        sim_ret, cam_quat = vrep.simxGetObjectQuaternion(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        cam_matrix = np.array(cam_matrix+[0,0,0,1]).reshape(4,4)

        # focal_length = 229.4516463256709
        focal_length = self.focal_length
        principal = self.principal
        extrinsic = calExtrinsic(cam_pos, cam_quat)
        intrinsic = [[focal_length, 0, principal],[0, focal_length, principal],[0, 0, 1]]
        mask = np.any(mask, axis=2)
        pcd = g.build_point_cloud(depth_img, intrinsic, extrinsic)
        grasping_candidates = g.get_grasping_candidates(pcd, mask)
        
        return grasping_candidates, pcd

    def execute_grasp(self, grasp_matrices):
        for grasp_matrix in grasp_matrices:       
            if (np.arccos(np.dot( grasp_matrix[:,0].reshape(1,3), np.array([0, -1, 0]).reshape(3,1) )) > np.radians(80) or
                np.arccos(np.dot( grasp_matrix[:,0].reshape(1,3), np.array([0, 0, -1]).reshape(3,1) )) > np.radians(80)):
                continue
            grasp_matrix = np.pad(grasp_matrix, ((0,1),(0,0)), mode='constant', constant_values=0.0) # (3, 4) -> (4, 4)
            grasp_matrix[3,3] = 1.0
            target_matrix = np.dot(grasp_matrix, self.original_matrix_gdn)
            
            send_matrix = list(target_matrix.flatten())[0:12]
            
            _,retInts,_,_,_ = \
                vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'RuntoTarget',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            running = True
            while running:
                _,retInts,_,_,_=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_TargetMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
                if retInts[0] == 0:
                    running = False
            return

    def get_intrinsic_matrix(self):
        intrinsic_matrix = np.array([[self.focal_length, 0, self.principal, 0],[0, self.focal_length, self.principal, 0],[0,0,1,0]],dtype=np.float64)
        #print('='*20+'\nintrinsic matrix')
        #print(intrinsic)
        return intrinsic_matrix

    def get_extrinsic_matrix(self, cam_handle):
        sim_ret, cam_pos = vrep.simxGetObjectPosition(self.clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        sim_ret, cam_quat = vrep.simxGetObjectQuaternion(self.clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        qx, qy, qz, qw = cam_quat
        tx, ty, tz = cam_pos
        extrinsic = [
            [1-2*(qy**2)-2*(qz**2), 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw, tx],
            [2*qx*qy+2*qz*qw, 1-2*(qx**2)-2*(qz**2), 2*qy*qz-2*qx*qw, ty],
            [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*(qx**2)-2*(qy**2), tz], 
            [0, 0, 0, 1]
        ]
        return extrinsic

    def get_obj_extrinsic(self, obj_handle):
        _, obj_pos = vrep.simxGetObjectPosition(self.clientID, obj_handle, -1, vrep.simx_opmode_oneshot_wait)
        _, obj_quat = vrep.simxGetObjectQuaternion(self.clientID, obj_handle, -1, vrep.simx_opmode_oneshot_wait)
        qx, qy, qz, qw = obj_quat
        tx, ty, tz = obj_pos
        extrinsic = [
            [1-2*(qy**2)-2*(qz**2), 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw, tx],
            [2*qx*qy+2*qz*qw, 1-2*(qx**2)-2*(qz**2), 2*qy*qz-2*qx*qw, ty],
            [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*(qx**2)-2*(qy**2), tz], 
            [0, 0, 0, 1]
        ]
        return extrinsic

    def get_im_uv(self, xyz, pixel_h=256, pixel_w=256):
        xyz = np.expand_dims(np.array(xyz), 1)
        intrinsic = self.get_intrinsic_matrix()
        extrinsic = self.get_extrinsic_matrix(self.cam_handle)
        extrinsic_inv = np.linalg.inv(extrinsic)
        xyz = np.matmul(extrinsic_inv, xyz)
        uvw = np.matmul(intrinsic, xyz)
        u, v, w = uvw[:3, 0]
        u = u / w
        v = v / w
        u = 2*self.principal - int(u)
        v = 2*self.principal - int(v)
        return u, v

    def get_boxes(self):
        obj_coordinates = {'cube':[-0.014459051191806793, 0.014459051191806793, -0.014533758163452148, 0.014533758163452148, -0.014728248119354248, 0.014728248119354248],
        'can':[-0.01691405475139618, 0.01691405475139618, -0.016965925693511963, 0.016965925693511963, -0.025441229343414307, 0.025441229343414307],
        'box':[-0.009191668592393398, 0.009191668592393398, -0.021037548780441284, 0.021037548780441284, -0.027341270819306374, 0.027341270819306374],
        'bottle':[-0.011533260345458984, 0.011533260345458984, -0.019278794527053833, 0.019278794527053833, -0.03827831894159317, 0.03827831894159317]}
        box_list = []
        for handle, cls in self.obj_list:
            paras = obj_coordinates[cls]
#         _,_,paras,_,_ = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getBoundingBox',[handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
#         print(paras)
#         if len(paras) != 6:
#             reset_flag = True
#             break
            _, pos = vrep.simxGetObjectPosition(self.clientID, handle, -1, vrep.simx_opmode_oneshot_wait)
            ext_mat = self.get_obj_extrinsic(handle)
            box_8 = np.array([ [paras[0], paras[2], paras[4], 1], [paras[1], paras[2], paras[4], 1], [paras[0], paras[3], paras[4], 1], [paras[1], paras[3], paras[4], 1], \
                        [paras[0], paras[2], paras[5], 1], [paras[1], paras[2], paras[5], 1], [paras[0], paras[3], paras[5], 1], [paras[1], paras[3], paras[5], 1]])
            
            box_8 = np.transpose(box_8)
            global_xyz = np.matmul(ext_mat, box_8)
            global_xyz = np.transpose(global_xyz)
            u_min = float('inf')
            u_max = 0
            v_min = float('inf')
            v_max = 0
            for i in range(8):
                u, v = self.get_im_uv(global_xyz[i])
                if u > u_max: u_max = u
                if u < u_min: u_min = u
                if v > v_max: v_max = v
                if v < v_min: v_min = v
            if u_max > (self.principal*2 - 1): u_max = self.principal*2 - 1
            if u_min < 0: u_min = 0
            if v_max > (self.principal*2 - 1): v_max = self.principal*2 - 1
            if v_min < 0: v_min = 0
            box = [u_min, v_min, u_max, v_max, self.cls2label[cls]]
            box_list.append(box)
        return box_list


    def clear_obj(self):
        for obj, _ in self.obj_list:
            vrep.simxRemoveModel(self.clientID, obj, vrep.simx_opmode_oneshot_wait) 
        self.obj_list = []

    def place_obj(self, cls):
        obj_filename = cls + '.ttm'
        _, obj = vrep.simxLoadModel(self.clientID, f'/home/tony/VREP-env/YCB/{obj_filename}', 0, vrep.simx_opmode_blocking)
        _x = random.uniform(-0.259+0.07, 0.01-0.07)
        _y = random.uniform(-0.585+0.07, -0.31-0.07)
        dropping_pos = self.cam_pos.copy()
        dropping_pos[0] = _x
        dropping_pos[1] = _y
        dropping_pos[2] = 0.5
        vrep.simxSetObjectPosition(self.clientID, obj, -1, dropping_pos, vrep.simx_opmode_blocking)
        return obj

    def settle_obj_replace(self, setting='cube'):
        self.clear_obj()
        list2shuffle = []
        if setting == 'cube':
            num = random.randint(1, 1)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('bottle')
        
        elif setting == 'can':
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(1, 1)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('bottle')

        elif setting == 'box':
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(1, 1)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('bottle')

        elif setting == 'bottle':
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(1, 2)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(1, 1)
            for i in range(num):
                list2shuffle.append('bottle')

        else:
            raise Exception('Invalid setting')
        random.shuffle(list2shuffle)
        for cls in list2shuffle:
            obj = self.place_obj(cls)
            self.obj_list.append((obj, cls))

    def settle_obj_dense(self, setting='cube'):
        self.clear_obj()
        list2shuffle = []
        if setting == 'cube':
            num = random.randint(20, 22)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('bottle')
        
        elif setting == 'can':
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(18, 20)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('bottle')

        elif setting == 'box':
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(16, 18)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('bottle')

        elif setting == 'bottle':
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(0, 1)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(16, 18)
            for i in range(num):
                list2shuffle.append('bottle')

        else:
            raise Exception('Invalid setting')
        random.shuffle(list2shuffle)
        for cls in list2shuffle:
            obj = self.place_obj(cls)
            self.obj_list.append((obj, cls))
        
    def settle_obj_particular_dense(self, setting='cube'):
        self.clear_obj()
        list2shuffle = []
        if setting == 'cube':
            num = random.randint(16, 18)
            for i in range(num):
                list2shuffle.append('cube')
        
        elif setting == 'can':
            num = random.randint(14, 16)
            for i in range(num):
                list2shuffle.append('can')

        elif setting == 'box':
            num = random.randint(8, 11)
            for i in range(num):
                list2shuffle.append('box')

        elif setting == 'bottle':
            num = random.randint(6, 8)
            for i in range(num):
                list2shuffle.append('bottle')
        else:
            raise Exception('Invalid setting')
        random.shuffle(list2shuffle)
        for cls in list2shuffle:
            obj = self.place_obj(cls)
            self.obj_list.append((obj, cls))

    def settle_obj(self, setting='cube'):
        self.clear_obj()
        list2shuffle = []
        if setting == 'cube':
            num = random.randint(10, 12)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('bottle')
        elif setting == 'can':
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(10, 12)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('bottle')
        elif setting == 'box':
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(8, 10)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('bottle')
        elif setting == 'bottle':
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('cube')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('can')
            num = random.randint(0, 2)
            for i in range(num):
                list2shuffle.append('box')
            num = random.randint(7, 9)
            for i in range(num):
                list2shuffle.append('bottle')
        else:
            raise Exception('Invalid setting')
        random.shuffle(list2shuffle)
        for cls in list2shuffle:
            self.obj_list.append((self.place_obj(cls), cls))

    def settle_particular_obj(self, setting='cube'):
        self.clear_obj()
        if setting == 'cube':
            for i in range(14):
                cls = 'cube'
                self.obj_list.append((self.place_obj(cls), cls))
        elif setting == 'can':
            for i in range(12):
                cls = 'can'
                self.obj_list.append((self.place_obj(cls), cls))
        elif setting == 'box':
            for i in range(10):
                cls = 'box'
                self.obj_list.append((self.place_obj(cls), cls))
        elif setting == 'bottle':
            for i in range(9):
                cls = 'bottle'
                self.obj_list.append((self.place_obj(cls), cls))
        else:
            raise Exception('Invalid setting')

    def query_taking(self, save_path):
        rgb = self.get_rgb(self.cam_handle)
        cv2.imwrite(save_path, rgb)
        boxes = self.get_bbox()
        print(len(boxes))

    def get_camera_matrix(self):  
        #res,retInts,self.robot_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[robot_handle],[],[],emptyBuff,vrep.simx_opmode_oneshot_wait)
        res,retInts,self.cam_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[self.cam_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_pos = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_quat = vrep.simxGetObjectQuaternion(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        self.cam_matrix = np.array(self.cam_matrix,dtype=np.float64).reshape(3,4)
        
        #print('='*20+'\ncam matrix')
        #print(cam_matrix)
        return self.cam_matrix
    
        
    def get_rgb(self, cam_handle):
        # RGB Info
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.clientID, cam_handle, 0, vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)
        color_img[color_img < 0] += 255
        color_img = cv2.flip(color_img,0).astype(np.uint8)
        color_img = cv2.cvtColor(color_img,cv2.COLOR_RGB2BGR)
        #print('='*20+'\ncolor_img')
        #print(color_img.shape)
        # cv2.imwrite('/home/tony/Desktop/vrep_im.jpg', color_img)
        return color_img

    def get_depth(self, Near_plane=0.02, Far_plane= 0.5):
        # Depth Info
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.cam_handle, vrep.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = cv2.flip(depth_img,0).astype(np.float)
        depth_img = (Far_plane - Near_plane) * depth_img + Near_plane
        #depth_img = depth_img * 1000
        #print('='*20+'\ndepth_img')
        #print(depth_img.shape)
        return depth_img.astype(np.float64)

    def get_bbox(self):
        color_img = self.get_rgb(self.cam_handle) 
        img_gray = cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)
        # plt.imshow(img_gray)
        # plt.show()
        ret,thresh = cv2.threshold(img_gray,200,255,0)
        # plt.imshow(thresh)
        # plt.show()
        thresh = 255 - thresh
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        boxes = []
        for i in range(len(contours)):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)    
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            min_h = float('inf')
            max_h = 0
            min_w = float('inf')
            max_w = 0
            for n in range(4):
                x = box[n, 0]
                y = box[n, 1]
                if x > max_w: max_w = x
                if x < min_w and x >= 0: min_w = x
                if y > max_h: max_h = y
                if y < min_h and y >= 0: min_h = y
        # print(f'{min_h}, {max_h}, {min_w}, {max_w}')
            if max_h - min_h < 16 or max_w - min_w <16:
                continue
            else:
                boxes.append([(min_w, min_h), (max_w, max_h)])
        #     cv2.rectangle(color_img, (min_w, min_h), (max_w, max_h), (0, 255, 0), 2)
        # plt.imshow(color_img)
        # plt.show()
        return boxes

    def get_mask(self, im, boxes):
        mask = np.zeros((im.shape[0], im.shape[1], 1))
        for i in range(len(boxes)):
            p1, p2 = (boxes[i][0], boxes[i][1])
            mask[p1[1]:p2[1], p1[0]:p2[0], :] = 1
        return mask

    def finish(self):
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot_wait)
        vrep.simxGetPingTime(self.clientID)
        vrep.simxFinish(self.clientID)
        print('Stop server!')