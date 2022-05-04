# This script generates/captured required images for the framework.
# 1. Color image
# 2. Color image (with projection)
# 3. Projection mask (2 - 1)
# 4. Color image (with checkerboard)
# 5. Depth image

import socket
import time
import socket
import os

import open3d as o3d
import cv2
import pymeshlab

import numpy as np
import imageio

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, pyqtSignal

from common import *

if __name__ == '__main__':

    res = 600

    grid = [int(i * (res/20)) for i in range(20)][1:]

    myapp = myImageDisplayApp()
    iPadCam = iPadCamera()

    imageio.plugins.freeimage.download()

    if not os.path.isdir(scene_path):
        os.mkdir(scene_path)
    os.chdir(scene_path)

    # Capture depth/color without projection
    myapp.emit_image_update('prj_texture_black.png')
    time.sleep(2)
    iPadCam.get_depth_image("depth")
    