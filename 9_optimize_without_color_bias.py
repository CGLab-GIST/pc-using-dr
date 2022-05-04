# This script optimizes optimal projector input image and theta


import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse, ParameterMap
from mitsuba.python.autodiff import render, write_bitmap, Adam

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import os

import thinplate as tps
from common import *

def img_to_tps_form(img):
    return torch.unsqueeze(torch.reshape(img, (600, 600,3)).permute(2,0,1), 0)
    
def tps_to_img_form(tps_img):
    return torch.flatten(torch.squeeze(tps_img).permute(1,2,0))

def tps_to_ek_arr(tps_img):
    return Float(tps_to_img_form(tps_img))

def ek_arr_to_tps(ek_arr):
    return img_to_tps_form(ek_arr.torch())
    

def clamp(arr, min=None, max=None):
    if min==None:
        min=arr
    if max==None:
        max=arr
    return ek.select(arr <= Float(max), ek.select(arr >= Float(min), arr, Float(min)), Float(max))

if __name__ == "__main__":

    myapp = myImageDisplayApp()
    # iPad = iPadCamera()

    os.chdir(scene_path)

    imageio.plugins.freeimage.download()    

    output_dir = 'result_wo_bias/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load example scene
    Thread.thread().file_resolver().append('.')
    scene = load_file('/out_scene.xml')

    params = traverse(scene)
    params.keep(['Projector.irradiance.data'])
    params.update()

    # Load TPS parameter (torch)
    theta = torch.load("tps_param.pt")
    # c_dst : normalized control points
    c_dst = tps.torch.uniform_grid((6,6)).view(-1, 2)
    grid = tps.torch.tps_grid(theta, torch.tensor(c_dst), torch.Size([1, 3, 600, 600])).cuda()

    # Load reference image
    ref = read_png("ref.png")

    # Set optimizer
    opt = Adam(params, lr=0.01)

    # Set initial bias
    bias = Float(np.zeros((640*480*3)))

    # Dummy reading..
    read_png("captured_check.png")

    time_a = time.time()
    
    ################################################
    # Optimize projector input image without theta #
    ################################################

    for it in range(501):
        
        # Constraint 1. Projected image is not a HDR image.
        params["Projector.irradiance.data"] = clamp(params['Projector.irradiance.data'], min=0.0, max=1.0)
        params.update()

        # Render image
        render_img = render(scene, optimizer=opt, unbiased=True, spp=5)
        render_img = clamp(render_img, min=0.0, max=1.0)

        # write_bitmap(output_dir+"rendered_%03i.png" % it, render_img, (640, 480))
        write_bitmap(output_dir+"proj_input_%03i.png" % it, params["Projector.irradiance.data"], (600, 600))
        
        loss = ek.hsum(ek.sqr(ref - (render_img))) / len(ref)
        ek.backward(loss)
        opt.step()

        if it%20 == 0:
            print("iter : ", it, "error : ", loss)


    time_b = time.time()
    print('total : %f' % (((time_b - time_a) * 1000)), 'ms')
    
    # project 200'th img
    # Constraint 1. Projected image is not a HDR image.
    params["Projector.irradiance.data"] = clamp(params['Projector.irradiance.data'], min=0.0, max=1.0)
    params.update()

    # Apply TPS
    warped = tps_to_ek_arr(F.grid_sample(ek_arr_to_tps(params["Projector.irradiance.data"]), grid))

    # Save texture
    write_bitmap(output_dir+"warped_texture_500.png", warped, (600, 600))
    time.sleep(5)
    myapp.emit_image_update(output_dir+"warped_texture_500.png")

    #############################################
    # Optimize projector input image with theta #
    #############################################

    time_a = time.time()


    time_b = time.time()
    print('total : %f' % (((time_b - time_a) * 1000)), 'ms')
