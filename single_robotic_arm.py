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
from matplotlib import pyplot as plt
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
        sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor", vrep.simx_opmode_oneshot_wait)
        sim_ret, self.shot_cam_handle = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor_1", vrep.simx_opmode_oneshot_wait)

        sim_ret, self.cam_pos = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        
        

        self.original_matrix = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],np.float)
        self.original_matrix_gdn = np.array([[-1,0,0,0],
                                             [0,-1,0,0],
                                             [0,0,1,0],
                                             [0,0,0,1]],np.float) 
        self.focal_length = 309.019
        self.principal = 128

        self.obj_filenames =  [i for i in os.listdir('YCB') if '.ttm' in i]
    
    def shot_taking(self, action_matrix):
        send_matrix = action_matrix
        res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'ShotMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
        running = True
        while running:
            res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_ShotMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False
    
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

        focal_length = 229.4516463256709
        principal = 128
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
    # def execute_grasp(self, grasp_matrices):
    #     for grasp_matrix in grasp_matrices:       
    #         if (np.arccos(np.dot( grasp_matrix[:,0].reshape(1,3), np.array([0, -1, 0]).reshape(3,1) )) > np.radians(80) or
    #             np.arccos(np.dot( grasp_matrix[:,0].reshape(1,3), np.array([0, 0, -1]).reshape(3,1) )) > np.radians(80)):
    #             continue
    #         grasp_matrix = np.pad(grasp_matrix, ((0,1),(0,0)), mode='constant', constant_values=0.0) # (3, 4) -> (4, 4)
    #         grasp_matrix[3,3] = 1.0
    #         target_matrix = np.dot(grasp_matrix, self.original_matrix_gdn)
            
    #         send_matrix = list(target_matrix.flatten())[0:12]
            
    #         _,retInts,_,_,_ = \
    #             vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'RuntoTarget',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
    #         running = True
    #         while running:
    #             _,retInts,_,_,_=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_TargetMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
    #             if retInts[0] == 0:
    #                 running = False

    def place_obj(self, n):
        dropping_pos = self.cam_pos.copy()
        dropping_pos[1] -= 0.2
        dropping_pos[2] -= 0.8
        obj_filename = self.obj_filenames[n]
        rc, obj = vrep.simxLoadModel(self.clientID, '/home/tony/VREP-env/YCB/%s' % obj_filename, 0, vrep.simx_opmode_oneshot_wait)
        vrep.simxScale
        vrep.simxSetObjectPosition(self.clientID, obj, -1, dropping_pos, vrep.simx_opmode_oneshot_wait)
        time.sleep(3)

    def get_camera_matrix(self):  
        #res,retInts,self.robot_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[robot_handle],[],[],emptyBuff,vrep.simx_opmode_oneshot_wait)
        res,retInts,self.cam_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[self.cam_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_pos = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        sim_ret, self.cam_quat = vrep.simxGetObjectQuaternion(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        self.cam_matrix = np.array(self.cam_matrix,dtype=np.float64).reshape(3,4)
        
        #print('='*20+'\ncam matrix')
        #print(cam_matrix)
        return self.cam_matrix
    
    def get_intrinsic_matrix(self):
        intrinsic_matrix = np.array([[self.focal_length,0, self.principal],[0, self.focal_length, self.principal],[0,0,1]],dtype=np.float64)
        #print('='*20+'\nintrinsic matrix')
        #print(intrinsic)
        return intrinsic_matrix
        
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
        ret,thresh = cv2.threshold(img_gray,200,255,0)
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
                if x < min_w: min_w = x
                if y > max_h: max_h = y
                if y < min_h: min_h = y
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

    # def get_mask(self, cam_handle):
    #     color_img = self.get_rgb(cam_handle) 
    #     img_gray = cv2.cvtColor(color_img,cv2.COLOR_RGB2GRAY)
    #     # print(img_gray)
    #     # plt.imshow(img_gray)
    #     # plt.show()
    #     ret,thresh = cv2.threshold(img_gray,200,255,0)
    #     # plt.imshow(thresh)
    #     # plt.show()
    #     thresh = 1 - thresh
    #     contours,hierarchy = cv2.findContours(thresh, 1, 2)
    #     masks = []
    #     for i in range(len(contours)-1):
    #         mask = np.zeros((color_img.shape[0],color_img.shape[1]), np.uint8)
    #         cv2.drawContours(mask,[contours[i]], -1,255,-1)
    #         masks.append(mask)
    #     return np.array(masks)

    def visualize_image(self, mask, depth_img, color_img):
        depth_img_show = depth_img - np.min(depth_img)
        depth_img_show /= np.max(depth_img_show)
        depth_img_show = (depth_img_show*255.0).astype(np.uint8)
        color_img_show = color_img.copy()
        # cv2.imshow("mask",mask)
        # cv2.waitKey(6000)

    def naive_grasp_detection(self, rgb_img, depth_img, show_bbox = False):

        img_gray = cv2.cvtColor(rgb_img,cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(img_gray,127,255,0)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)

        rgb_img_show = rgb_img.copy()

        grasp_list = []
        for i in range(len(contours)-1):
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)    
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(rgb_img_show,[box],0,(0,0,255),2)
            
            pos_x = int((box[0][0]+box[2][0])/2)
            pos_y = int((box[0][1]+box[2][1])/2)
            pos_z = np.min(depth_img)
            angle = -1*rect[2]/180*math.pi

            z_offset = 0.02
            pos_x_cam = -1*pos_z*(pos_x-self.principal)/self.focal_length
            pos_y_cam =-1*pos_z*(pos_y-self.principal)/self.focal_length
            pos_cam = np.array([pos_x_cam,pos_y_cam,pos_z,1.0],np.float64).reshape(4,1)
            pos_world = np.dot(self.cam_matrix,pos_cam)
            pos_world[2,0] = pos_world[2,0] - z_offset

            grasp_matrix = np.array([[math.cos(angle),-math.sin(angle),0,pos_world[0,0]],
                    [math.sin(angle),math.cos(angle),0,pos_world[1,0]],
                    [0,0,1,pos_world[2,0]],
                    [0,0,0,1]],np.float64)
            grasp_list.append(grasp_matrix)

        if show_bbox:
            cv2.imshow('bbox',rgb_img_show)
            cv2.waitKey(1000)
        
        return grasp_list

    def run_grasp(self, grasp_list, num, use_gdn = True):
        
        if len(grasp_list) == 0:
            print('No any grasp detected!')
        grasp_iter = min(num, len(grasp_list))
        for i in range(grasp_iter):
            if use_gdn:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix_gdn)
            else:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix)
            send_matrix = list(grasp_matrix.flatten())[0:12]
            res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'GraspMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
            running = True
            while running:
                res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_GraspMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
                if retInts[0] == 0:
                    running = False

    def get_target_matrix(self, obj_name):
        
        sim_ret, self.target_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        res,retInts,self.target_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[self.target_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        self.target_matrix = np.array(self.target_matrix+[0,0,0,1],dtype=np.float64).reshape(4,4)
        self.target_matrix[2,3] = self.target_matrix[2,3] + 0.05

    def get_correct_action_matrix(self, obj_name):

        sim_ret, self.obj_handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_oneshot_wait)
        sim_ret, self.obj_pos = vrep.simxGetObjectPosition(self.clientID, self.obj_handle, -1, vrep.simx_opmode_oneshot_wait)
        sim_ret, self.obj_quat = vrep.simxGetObjectQuaternion(self.clientID, self.obj_handle, -1, vrep.simx_opmode_oneshot_wait)
        
        res,retInts,self.obj_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[self.obj_handle],[],[],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
        self.obj_matrix = np.array(self.obj_matrix+[0,0,0,1],dtype=np.float64).reshape(4,4)
        self.obj_matrix[2,3] = self.obj_matrix[2,3] - 0.0493

        # target_matrix = action_matrix * obj_matrix
        action_matrix = np.dot(self.target_matrix,np.linalg.inv(self.obj_matrix))

        return action_matrix


    def pick_and_place(self, grasp_list, num, action_matrix, use_gdn=True):

        if len(grasp_list) == 0:
            print('No any grasp detected!')
        grasp_iter = min(num, len(grasp_list))
        for i in range(grasp_iter):
            # grasping
            if use_gdn:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix_gdn)
            else:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix)
            send_matrix = list(grasp_matrix.flatten())[0:12]
            res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'GraspMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
            running = True
            while running:
                res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_GraspMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
                if retInts[0] == 0:
                    running = False

            # taking shot
            # shot_matrix = self.get_shot_matrix(-7.486, -18.404, 143.26)
            # send_matrix = list(shot_matrix.flatten())[0:12]
            # res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'ShotMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
            # running = True
            # while running:
            #     res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_ShotMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
            #     if retInts[0] == 0:
            #         running = False
            send_matrix = [250*math.pi/180,-20*math.pi/180,110.01*math.pi/180,-15*math.pi/180,-90*math.pi/180,90*math.pi/180]
            res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'ShotMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
            running = True
            while running:
                res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_ShotMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
                if retInts[0] == 0:
                    running = False

            
            sim_ret = vrep.simxSetObjectPosition(self.clientID, self.obj_handle, -1, self.obj_pos, vrep.simx_opmode_oneshot_wait)
            sim_ret = vrep.simxSetObjectQuaternion(self.clientID, self.obj_handle, -1, self.obj_quat, vrep.simx_opmode_oneshot_wait)

    def eulerAngle2rotationMatrix(self, alpha, beta, gamma):
        yaw = np.array([[math.cos(gamma),-math.sin(gamma),0],
                        [math.sin(gamma),math.cos(gamma),0],
                        [0,0,1]])
        pitch = np.array([[math.cos(beta), 0, math.sin(beta)],
                        [0, 1, 0], 
                        [-math.sin(beta), 0, math.cos(beta)]])
        roll = np.array([[1, 0, 0], 
                        [0, math.cos(alpha), -math.sin(alpha)],   
                        [0, math.sin(alpha), math.cos(alpha)]])
        rotation_matrix = np.dot(np.dot(yaw, pitch), roll)
        return rotation_matrix
        
    def get_shot_matrix(self, alpha, beta, gamma, x=0.37568, y=-0.37313, z=0.42084):
        rotation_matrix = self.eulerAngle2rotationMatrix(alpha*math.pi/180, beta*math.pi/180, gamma*math.pi/180)
        shot_matrix = np.zeros((4, 4))
        shot_matrix[0, 3] = x
        shot_matrix[1, 3] = y
        shot_matrix[2, 3] = z
        shot_matrix[3, 3] = 1.
        shot_matrix[0, 0:3] = rotation_matrix[0, 0:3]
        shot_matrix[1, 0:3] = rotation_matrix[1, 0:3]
        shot_matrix[2, 0:3] = rotation_matrix[2, 0:3]
        return shot_matrix

    def finish(self):
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot_wait)
        vrep.simxGetPingTime(self.clientID)
        vrep.simxFinish(self.clientID)
        print('Stop server!')

    def object_grasping(self, grasp_list, num, use_gdn=True):
        if len(grasp_list) == 0:
            print('No any grasp detected!')
        grasp_iter = min(num, len(grasp_list))
        for i in range(grasp_iter):
            # grasping
            if use_gdn:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix_gdn)
            else:
                grasp_matrix = np.dot(grasp_list[i], self.original_matrix)
            send_matrix = list(grasp_matrix.flatten())[0:12]
            res,retInts,path,retStrings,retBuffer = vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'GraspMovement',[],send_matrix,[],self.emptyBuff,vrep.simx_opmode_oneshot_wait) 
            running = True
            while running:
                res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(self.clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],['UR5_GraspMatrix'],self.emptyBuff,vrep.simx_opmode_oneshot_wait)
                if retInts[0] == 0:
                    running = False