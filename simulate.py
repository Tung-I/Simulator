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
from pkg_resources import parse_version
import gym
from bullet.tm700 import tm700
from bullet.tm700_possensor_Gym import tm700_possensor_gym
from bullet.tm700_rgbd_Gym import tm700_rgbd_gym


def get_name_to_link(model_id):
    link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'):-1,}
    for _id in range(p.getNumJoints(model_id)):
        _name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')
        link_name_to_index[_name] = _id
    return link_name_to_index


if __name__ == '__main__':
    p.connect(p.GUI)
    test = tm700_rgbd_gym(width=720, height=720, numObjects=7, objRoot='../YCB_valset_urdf')
    for ite in range(1000):
        test.reset()
        tm_link_name_to_index = get_name_to_link(test._tm700.tm700Uid)
        table_link_name_to_index = get_name_to_link(test.tableUid)
        obj_link_name_to_index = []
        for uid in test._objectUids:
            obj_link_name_to_index.append((uid, get_name_to_link(uid)))