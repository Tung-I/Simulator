import random
import os
from gym import spaces
import time
import json
import pybullet as p
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pathlib import Path
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from bullet.tm700_possensor_Gym import tm700_possensor_gym

DATA_ROOT = '/home/tony/datasets/ShapeNet_3obj'

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
    random.seed(0)
    root = Path(DATA_ROOT)
    class_dirs = [c for c in root.iterdir() if c.is_dir() and 'ipynb' not in str(c)]
    obj_list = []
    obj_name_list = []
    for c_d in class_dirs:
        obj_name_list.append(c_d.parts[-1])
        obj_dirs = [sub_dir for sub_dir in c_d.iterdir() if sub_dir.is_dir()]
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

    self.table_pose = [0.5000000, 0.00000, -.640000, 0.000000, 0.000000, 0.0, 1.0]
    self.tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), *self.table_pose)

    p.setGravity(0, 0, -10)
    self._tm700 = tm700(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

    self._envStepCounter = 0
    p.stepSimulation()

    self._objectUids = self._randomly_place_objects(self._objList)
    self._observation = self._get_observation()
    return np.array(self._observation)

  def _randomly_place_objects(self, objList):
    """Randomly places the objects in the bin.
    """
    objectUids = []
    shift = [0, -0.02, 0]
    meshScale = [0.2, 0.2, 0.2]
    list_x_pos = [0.55, 0.6, 0.65]
    list_y_pos = [0.2, 0.05, 0.2]

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
                        basePosition=[xpos, ypos, .15],
                        useMaximalCoordinates=True)

        objectUids.append(uid)
        for _ in range(1000):
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

    return test



  