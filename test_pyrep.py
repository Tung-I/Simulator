import numpy as np
from pyrep.backend import sim
from pyrep.const import PerspectiveMode, RenderMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep import PyRep

def build_point_cloud(depth_img, intrinsic, trans_mat):
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


SCENE_FILE = 'test_grasp.ttt'
pr = PyRep()

pr.launch(SCENE_FILE, headless=False)
pr.start()

sensor = VisionSensor('Vision_sensor')
trans_mat = sim.simGetObjectMatrix(sensor.get_handle(), -1) #list, [12]
trans_mat = np.array(trans_mat).reshape(3, 4)
intrinsic = [[229.4516463256709, 0, 128], [0, 229.4516463256709, 128], [0, 0, 1]]


sensor1 = VisionSensor('Vision_sensor_61')
depth_img = sensor1.capture_depth()
print(trans_mat)

print(depth_img.shape)
np.save('depth.npy', depth_img)

pcd = build_point_cloud(depth_img, intrinsic, trans_mat).astype(np.float32)
np.save('dump.npy', pcd.reshape(-1, 3))

print('Done ...')
input('Press enter to finish ...')
pr.stop()
pr.shutdown()
