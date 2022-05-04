# This script optimizes M and W

import time
import os
import shutil


import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, render_torch, write_bitmap, Adam

import thinplate as tps

from common import *

def img_to_tps_form(img):
    return torch.unsqueeze(torch.reshape(img, (600, 600,3)).permute(2,0,1), 0)
    
def tps_to_img_form(tps_img):
    return torch.flatten(torch.squeeze(tps_img).permute(1,2,0))

def tps_to_ek_arr(tps_img):
    return Float(tps_to_img_form(tps_img))


if __name__ == "__main__":

    imageio.plugins.freeimage.download()

    os.chdir(scene_path)

    # Optimize M
    output_dir = 'tps_inverse_result/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load example scene
    Thread.thread().file_resolver().append('.')
    scene = load_file('/out_scene_for_pose.xml',  integrator="flag")

    params = traverse(scene)
    params.keep(['Projector.irradiance.data'])
    params['Projector.irradiance.data'] = read_png("prj_texture_pattern.png", 600, 600)
    params.update()
    params_torch = params.torch()

    # Load camera-captured image
    cam_img_torch = torch.reshape(read_png("captured_check.png").torch(), (480, 640, 3))
    
    cam_img_torch = torch.where(cam_img_torch > 0.7, torch.ones((480, 640, 3)).cuda(), torch.zeros((480, 640, 3)).cuda())
    write_bitmap(output_dir + "captured_check_mask.png", cam_img_torch, (640, 480))

    #################
    #   Optimize M  #
    #################

    # c_dst : normalized control points
    c_dst = tps.torch.uniform_grid((6,6)).view(-1, 2)
    # theta : zero array size of number of 1 * (control points + 2) * 2
    theta = torch.zeros(1, (c_dst.shape[0]+2), 2, requires_grad=True)
    
    # Put TPS parameter into optimizer instead of params_torch
    opt = torch.optim.Adam([theta], lr = 0.000005)

    tps_inv_param = img_to_tps_form(read_png("prj_texture_pattern.png", 600, 600).torch())

    time_a = time.time()

    for it in range(3000):
        
        opt.zero_grad()
        
        # Apply TPS
        warped = F.grid_sample(tps_inv_param, tps.torch.tps_grid(theta, torch.tensor(c_dst), tps_inv_param.shape).cuda())

        params_torch['Projector.irradiance.data'] = tps_to_img_form(warped)

        params['Projector.irradiance.data'] = Float(params_torch['Projector.irradiance.data'])
        params.update()

        render_img = render_torch(scene, params=params, unbiased=True, spp=20, **params_torch)

        mse_val = F.mse_loss(cam_img_torch, render_img)

        if it%20 == 0:
            write_bitmap(output_dir+"rendered_%03i.png"%it, render_img, (640, 480))
            write_bitmap(output_dir+"texture_%03i.png"%it, params['Projector.irradiance.data'], (600, 600))

        loss = mse_val

        loss.backward()
        opt.step()

        if it%20 == 0:
            print("iter : ", it, "error : ", loss.item())

    time_b = time.time()
    print('total : %f' % (((time_b - time_a))), 's')

    write_bitmap("tps_input_invtps_output.png", params['Projector.irradiance.data'], (600, 600))


    #################
    #   Optimize W  #
    #################

    print("Optimizing W...")
    output_dir = 'tps_result/'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Dummy reading function : Error occurs if removed... why?
    read_png("prj_texture_pattern.png", 600, 600)
    
    # input of tps (output of inverse tps)
    src = img_to_tps_form(read_png("tps_input_invtps_output.png", 600, 600).torch())
    # output of tps (input of inverse tps)
    target = img_to_tps_form(read_png("prj_texture_pattern.png", 600, 600).torch())

    # c_dst : normalized control points
    c_dst = tps.torch.uniform_grid((6,6)).view(-1, 2)
    # theta : zero array size of number of 1 * (control points + 2) * 2
    theta = torch.zeros(1, (c_dst.shape[0]+2), 2, requires_grad=True)
    
    opt = torch.optim.Adam([theta], lr = 0.00003)

    time_a = time.time()

    for i in range(3000):
        opt.zero_grad()

        grid = tps.torch.tps_grid(theta, c_dst.clone().detach(), src.shape).cuda()
        warped = F.grid_sample(src, grid).cuda()
        
        mse_val = F.mse_loss(warped, target)

        loss = mse_val
        loss.backward()
        opt.step()

        if i%20 == 0:
            write_bitmap(output_dir + "warped_%03i.png"%i, tps_to_img_form(warped), (600, 600))
            print(i, loss.item())

    time_b = time.time()
    print('total : %f' % (((time_b - time_a) * 1)), 's')
    
    # Save W parameter
    torch.save(theta, "tps_param.pt")
    
    myapp = myImageDisplayApp()
    iPad = iPadCamera()

    
    # To validate align with reference image
    target = read_png("prj_texture_pattern.png", 600, 600).torch()
    grid = tps.torch.tps_grid(theta, c_dst.clone().detach(), src.shape).cuda()
    warped = tps_to_img_form(F.grid_sample(img_to_tps_form(target), grid).cuda())
    write_bitmap("before_tps.png", target, (600, 600))
    write_bitmap("after_tps.png", warped, (600, 600))

    myapp.emit_image_update("before_tps.png")
    time.sleep(0.5)
    iPad.get_rgb_image("before_tps_result")

    time.sleep(0.5)

    myapp.emit_image_update("after_tps.png")
    time.sleep(0.5)
    iPad.get_rgb_image("after_tps_result")
