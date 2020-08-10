import random
import copy
import cv2
import numpy as np
from pathlib import Path 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm_notebook as tqdm


def get_img_xy(xyz, mat_view, mat_proj, pixel_h, pixel_w):
    xyz = np.concatenate([xyz, np.asarray([1.])], axis=0)
    mat_view = np.asarray(mat_view).reshape(4, 4)
    mat_proj = np.asarray(mat_proj).reshape(4, 4)
    xyz = np.dot(xyz, mat_view)
    xyz = np.dot(xyz, mat_proj)
    u, v, z = xyz[:3]
    u = u  / z  * (pixel_w / 2) + (pixel_w / 2)
    v = (1 - v / z)  *  pixel_h / 2
    return u, v

def get_2d_bbox(aabb, mat_view, mat_proj, pixel_h=320, pixel_w=320):
    x_min, y_min, z_min = aabb[0]
    x_max, y_max, z_max = aabb[1]
    top = float('inf')
    bot = 0
    left = float('inf')
    right = 0
    for _x in [x_min, x_max]:
        for _y in [y_min, y_max]:
            for _z in [z_min, z_max]:
                xyz = [_x, _y, _z]
                u, v = get_img_xy(xyz, mat_view, mat_proj, pixel_h, pixel_w)
                if u > right:
                    right = u if u < pixel_w else pixel_w - 1
                if u < left:
                    left = u if u >=0 else 0
                if v > bot:
                    bot = v if v < pixel_h else pixel_h - 1
                if v < top:
                    top = v if v >= 0 else 0
    w_bbox = right - left
    h_bbox = bot - top
    return left, top, right, bot

def vis_detections(im, dets):
    """Visual debugging of detections."""
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
        cv2.putText(im, IDX2NAME[int(dets[i, 4])], (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=2)
    return im

im = np.load('/home/tony/Desktop/obj_save/img_0.npy')
AABB = np.load('/home/tony/Desktop/obj_save/AABB_0.npy')
view_mat = np.load('/home/tony/Desktop/obj_save/view_mat_0.npy')
proj_mat = np.load('/home/tony/Desktop/obj_save/proj_mat_0.npy')

def _get_observation(self, unseen=False):
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
        segmentation = img_arr[4]
        depth = np.reshape(depth, (self._height, self._width,1) )
        segmentation = np.reshape(segmentation, (self._height, self._width,1) )

        np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        np_img_arr = np_img_arr[:, :, :3].astype(np.float64)

        view_mat = np.asarray(self._view_matrix).reshape(4, 4)
        proj_mat = np.asarray(self._proj_matrix).reshape(4, 4)
        # pos = np.reshape(np.asarray(list(p.getBasePositionAndOrientation(self._objectUids[0])[0])+[1]), (4, 1))

        AABBs = np.zeros((len(self._objectUids), 2, 3))
        cls_ls = []
        
        for i, (_uid, _cls) in enumerate(zip(self._objectUids, self._objectClasses)):
            AABBs[i] = np.asarray(p.getAABB(_uid)).reshape(2, 3)
            cls_ls.append(NAME2IDX[_cls])

        np.save('/home/tony/Desktop/obj_save/view_mat_'+str(self.img_save_cnt), view_mat)
        np.save('/home/tony/Desktop/obj_save/proj_mat_'+str(self.img_save_cnt), proj_mat)
        np.save('/home/tony/Desktop/obj_save/img_'+str(self.img_save_cnt), np_img_arr.astype(np.int16))
        np.save('/home/tony/Desktop/obj_save/AABB_'+str(self.img_save_cnt), AABBs)
        np.save('/home/tony/Desktop/obj_save/class_'+str(self.img_save_cnt), np.array(cls_ls))

        dets = np.zeros((AABBs.shape[0], 5))
        for i in range(AABBs.shape[0]):
            dets[i, :4] = self.get_2d_bbox(AABBs[i], view_mat, proj_mat, IM_HEIGHT, IM_WIDTH)
            dets[i, 4] = int(cls_ls[i])
        np.save(OUTPUT_DIR + '/annotation_'+str(self.img_save_cnt), dets)
        test = np.concatenate([np_img_arr[:, :, 0:2], segmentation], axis=-1)
        return test