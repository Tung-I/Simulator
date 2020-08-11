import random
import os
from gym import spaces
import pybullet as p
import numpy as np
import pybullet_data
from pathlib import Path
from bullet.tm700 import tm700
from bullet.tm700_possensor_Gym import tm700_possensor_gym


cwd = os.getcwd()
shapenet_dataset_folder_name = 'shapenet_subset'
DATA_ROOT = os.path.join(cwd, shapenet_dataset_folder_name)
dataset_path = 'datasets/simulated_data'

#create dataset class
class_name_list = os.listdir(shapenet_dataset_folder_name)
NAME2IDX = {}
for idx,class_name in enumerate(class_name_list):
    NAME2IDX[class_name] = idx+1
print('Category:',NAME2IDX)

class tm700_rgbd_gym_generate_simulated_data(tm700_possensor_gym):
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
        self.get_shot_random_seed = 6

        #dataset related
        self.cnt = 1


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


    def _get_random_obj(self):
        """Get 2~4 different obj from shapenet model
        """
        root = Path(DATA_ROOT)
        class_dirs = [c for c in root.iterdir() if c.is_dir() and 'ipynb' not in str(c)]
        obj_list = []
        obj_name_list = []
        random.seed()
        num = random.randint(2,4)
        random.seed()
        class_dirs = random.sample(class_dirs, k=num)

        for c_d in class_dirs:
            obj_name_list.append(c_d.parts[-1])
            obj_dirs = [sub_dir for sub_dir in c_d.iterdir() if sub_dir.is_dir()]
            random.seed()
            _dir = random.sample(obj_dirs, k=1)[0]
            obj_path = str(_dir / Path('models/model_normalized.obj'))
            obj_list.append(obj_path)
        return obj_list, obj_name_list

    def reset(self):
        """Environment reset called at the beginning of an episode.
        """

        # Set the camera settings.
        look = [0.45, 1.5, 0.7]
        self._cam_pos = look
        distance = 0.7
        pitch = -90
        yaw = 0
        roll = 0

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
        self.table_pose = [.5000000, 1.50000, -.640000, 0.000000, 0.000000, 0.0, 1.0]
        self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), *self.table_pose)
        #p.changeVisualShape(objectUniqueId=self.tableUid, linkIndex=-1, rgbaColor=[0, 0, 0, 0])
        p.setGravity(0, 0, -10)
        self._tm700 = tm700(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()

        # get .obj files paths and their classes
        self._objList, self._objNameList = self._get_random_obj()

        self._objectUids = self._randomly_place_objects(self._objList)
        self._get_observation()
        self.bbox_list = self.compute_pixel_coordinate()
        self.obj_anno_save()

        return np.array(self._observation)

    def _randomly_place_objects(self, objList):
        """Randomly places the objects in the bin.
        """
        objectUids = []
        shift = [0, -0.02, 0]
        meshScale = [0.2,0.2,0.2]
        #list_x_pos = [0.55, 0.6, 0.65]
        #list_y_pos = [0.2, 0.05, 0.2]
        list_x_pos = [random.randint(35,60)/100 for i in range(len(objList))]
        list_y_pos = [random.randint(150,170)/100 for i in range(len(objList))]
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
        np.save(os.path.join(dataset_path,'np_img/image_') + str(self.cnt), np_img_arr[:,:,:3])
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

    def get_img_xy(self,xyz, mat_view, mat_proj, pixel_h, pixel_w):
        xyz = np.concatenate([xyz, np.asarray([1.])], axis=0)
        mat_view = np.asarray(mat_view).reshape(4, 4)
        mat_proj = np.asarray(mat_proj).reshape(4, 4)
        xyz = np.dot(xyz, mat_view)
        xyz = np.dot(xyz, mat_proj)
        u, v, z = xyz[:3]
        u = u / z * (pixel_w / 2) + (pixel_w / 2)
        v = (1 - v / z) * pixel_h / 2
        return u, v

    def get_2d_bbox(self,aabb, mat_view, mat_proj, pixel_h=320, pixel_w=320):
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
                    u, v = self.get_img_xy(xyz, mat_view, mat_proj, pixel_h, pixel_w)
                    if u > right:
                        right = u if u < pixel_w else pixel_w - 1
                    if u < left:
                        left = u if u >= 0 else 0
                    if v > bot:
                        bot = v if v < pixel_h else pixel_h - 1
                    if v < top:
                        top = v if v >= 0 else 0
        w_bbox = right - left
        h_bbox = bot - top
        return left, top, w_bbox, h_bbox

    def compute_pixel_coordinate(self):
        bbox_list = []
        for uid in self._objectUids:
            aabb = list(p.getAABB(uid))
            bbox = list(self.get_2d_bbox(aabb,self._view_matrix,self._proj_matrix,self._width,self._height))
            bbox_list.append(bbox)
        return bbox_list


    def obj_anno_save(self):
        obj_info = []
        for idx, obj_name in enumerate(self._objNameList):
            obj_class_id = NAME2IDX[obj_name]
            bbox = self.bbox_list[idx]
            obj_info.append(np.concatenate([bbox,[obj_class_id]]))
        np.save(os.path.join(dataset_path,'np_anno/annotation_') + str(self.cnt) ,obj_info)
        self.cnt += 1

