# `BSD 3-Clause License

# Copyright (c) 2022, GIST CGLAB
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This script generates/captured required images for the framework.
# 1. Color image (for texture)
# 2. Color image (with white image projection)
# 3. Projection mask (2 - 1)
# 4. Color image (with checkerboard image projection)

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
