# This script generates target image by transforming images in /reference directory.

import os
import shutil

from common import *
from PIL import Image
import numpy as np
import argparse

def calc_offset(x1, y1, x2, y2):
    return x1, y1, x2-x1, y2-y1

if __name__ == "__main__":

    result_width = 640
    result_height = 480
    src_path = "./reference/"
    dst_path = os.path.join(src_path, "generated_test_imgs")
    
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path, exist_ok=True)

    offset_x, offset_y, transformed_width, transformed_height = calc_offset(184,53,184+310,53+310)


    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    file_list = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]

    for filename in file_list:
        
        src_file_path = os.path.join(src_path, filename)
        dst_file_path = os.path.join(dst_path, filename)

        img = Image.open(src_file_path)

        target_arr = np.zeros((result_height, result_width, 3))

        transformed_img = img.resize((transformed_width, transformed_height))
        img_arr = np.asarray(transformed_img)

        for i in range(transformed_width):
            for j in range(transformed_height):
                target_arr[j + offset_y][i + offset_x] = img_arr[j][i]


        if filename == "img_pattern.png":
            target_arr = np.where(target_arr > np.ones(target_arr.shape)*255*0.9, np.ones(target_arr.shape)*255, np.zeros(target_arr.shape))
        target_img = Image.fromarray(np.uint8(target_arr))

        target_img.save(dst_file_path, "PNG")

    shutil.copytree("./reference/generated_test_imgs", scene_path+"/dist_imgs")
    shutil.rmtree("./reference/generated_test_imgs")
