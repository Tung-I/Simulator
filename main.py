# simExtRemoteApiStart(19999)
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
import argparse
import os 
import os.path as osp
import json
from matplotlib import pyplot as plt
from datetime import datetime
from utils import *

print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections

while True:
    clientID = vrep.simxStart('127.0.0.1',19997,True,True,-500000,5) # Connect to V-REP
    if clientID != -1:
        print ('Connected to remote API server')
        returnCode = vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
        break
    else:
        print ('Failed connecting to remote API server')

time.sleep(0.5)
emptyBuff = bytearray()


target_box_pos = {'xmin': 0.001, 'xmax':0.4, 'ymin': -0.6, 'ymax': -0.3}
original_matrix = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],np.float)
original_matrix_gdn = np.array([[-1,0,0,0],
                                [0,-1,0,0],
                                [0,0,1,0],
                                [0,0,0,1]],np.float)


def execute_grasp(grasp_matrices):
    for grasp_matrix in grasp_matrices:
        
        if (np.arccos(np.dot( grasp_matrix[:,0].reshape(1,3), np.array([0, -1, 0]).reshape(3,1) )) > np.radians(80) or
            np.arccos(np.dot( grasp_matrix[:,0].reshape(1,3), np.array([0, 0, -1]).reshape(3,1) )) > np.radians(80)):
            continue
        grasp_matrix = np.pad(grasp_matrix, ((0,1),(0,0)), mode='constant', constant_values=0.0) # (3, 4) -> (4, 4)
        grasp_matrix[3,3] = 1.0
        target_matrix = np.dot(grasp_matrix, original_matrix_gdn)
        
        send_matrix = list(target_matrix.flatten())[0:12]
        
        _,retInts,_,_,_ = vrep.simxCallScriptFunction(clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'RuntoTarget',[],send_matrix,[],emptyBuff,vrep.simx_opmode_oneshot_wait)
        
        running = True
        while running:
            _,retInts,_,_,_=vrep.simxCallScriptFunction(clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'isRunning',[],[],[],emptyBuff,vrep.simx_opmode_oneshot_wait)
            if retInts[0] == 0:
                running = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    
    sim_ret, robot_handle = vrep.simxGetObjectHandle(clientID, "UR5", vrep.simx_opmode_oneshot_wait)
    sim_ret, cam_handle = vrep.simxGetObjectHandle(clientID, "Vision_sensor", vrep.simx_opmode_oneshot_wait)

    res,retInts,robot_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[robot_handle],[],[],emptyBuff,vrep.simx_opmode_oneshot_wait)
    res,retInts,cam_matrix,retStrings,retBuffer=vrep.simxCallScriptFunction(clientID,'RemoteAPI',vrep.sim_scripttype_childscript,'getObjectPose',[cam_handle],[],[],emptyBuff,vrep.simx_opmode_oneshot_wait)
    sim_ret, cam_pos = vrep.simxGetObjectPosition(clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)
    sim_ret, cam_quat = vrep.simxGetObjectQuaternion(clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)
    cam_matrix = np.array(cam_matrix+[0,0,0,1]).reshape(4,4)

    color_all = getRGB(clientID, cam_handle)
    depth_all = getDepth(clientID, cam_handle)

    # TODO
    plt.imshow(color_all)
    plt.show()
    raise Exception(' ')
    mask = model(color_all)

    raise Exception(' ')

    grasp_matrices, pcd = graspGDN(mask=mask, depth_img=depth_all, cam_pos=cam_pos, cam_quat=cam_quat, cam_mat=cam_matrix)
    
    execute_grasp(grasp_matrices)
    
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    vrep.simxGetPingTime(clientID)
    vrep.simxFinish(clientID)