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


    # Generate black image with given resolution
    cv2.imwrite("prj_texture_black.png", np.zeros((res, res, 3))*255)
    cv2.imwrite("prj_texture_white.png", np.ones((res, res, 3))*255)

    # Generate pattern image
    pattern_np = np.ones((res, res, 3))*255
    for i in grid:
        for j in range(-2, 3):
            pattern_np[i+j, :, :] = 0
    for i in grid:
        for j in range(-2, 3):
            pattern_np[:, i+j, :] = 0
    cv2.imwrite("prj_texture_pattern.png", pattern_np)

    input("Turn on the light and press enter")

    # Capture depth/color without projection
    myapp.emit_image_update('prj_texture_black.png')
    time.sleep(2)
    iPadCam.get_rgb_image("color")

    # Capture depth/color with projection
    myapp.emit_image_update('prj_texture_white.png')
    time.sleep(5)
    iPadCam.get_rgb_image("color_proj")

    # Generate mask 
    cam_img = read_png("color.png")
    cam_img_proj = read_png("color_proj.png")

    diff = cam_img_proj - cam_img
    diff = np.where(diff>0.01, 1, 0)
    diff = np.reshape(diff, (480, 640, 3))
    diff = np.mean(diff, axis=2)
    diff = np.reshape(diff, (480, 640, 1))
    diff = np.concatenate((diff, diff, diff), axis=2)

    pyexr.write("color_proj_dist.exr", diff)

    input("Turn off the light and press enter")

    time.sleep(2)

    # Project estimated projector input image
    print("Project check image")
    myapp.emit_image_update('prj_texture_pattern.png')
    time.sleep(5)
    
    # Capture image
    print("Capture check image")
    iPadCam.get_rgb_image("captured_check")
