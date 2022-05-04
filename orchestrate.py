import os
import glob
import shutil
import time

if __name__ == "__main__":

    dir = "data/plane/"
    dist_img_dir = dir + "dist_imgs/"


    for dist_img in os.listdir(dist_img_dir):
        if dist_img == "img_pattern.png":
            break

        shutil.copy(dist_img_dir+dist_img, dir+"ref.png")

        time.sleep(1)

        result_dir_path = dir+dist_img[:-4]+"_result/"
        os.mkdir(result_dir_path)
        
        os.system("python .\\5_3_optimize_bias_proj_img.py")
        shutil.move(dir+"result", result_dir_path+"result")
        
        if dist_img == "img_0003.png":
            os.system("python .\\6_1_optimize_without_warp.py")
            os.system("python .\\6_2_optimize_without_color_bias.py")
            shutil.move(dir+"result_without_warp", result_dir_path+"result_without_warp")
            shutil.move(dir+"result_without_bias", result_dir_path+"result_without_bias")

        shutil.move(dir+"ref.png", result_dir_path+"ref.png")

        