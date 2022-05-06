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

# This script optimizes projector's position and direction using differentiable rendering.

import os
import time

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, Thread
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, write_bitmap, Adam
import time
import os

import pyexr
import numpy as np

import matplotlib.pyplot as plt
import pyexr
import numpy as np

from mitsuba.core import xml, Thread, Transform4f, ScalarPoint3f, Bitmap, Float, Vector3f, UInt32, Point3f
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, render_torch, write_bitmap, Adam, SGD


from common import *


def projector_pose_template(pos, rot):
    return \
        "<?xml version='1.0' encoding='utf-8'?>\n"+\
        "<scene version=\"0.5.0\">\n"+\
        "    <point name=\"position\" value=\""+str(pos)[2:-2]+"\"/>\n"+\
		"    <vector name=\"rotation\" value=\""+str(rot)[2:-2]+"\"/>\n"+\
        "</scene>\n"


if __name__ == "__main__":

    os.chdir(scene_path)

    # Prepare output folder
    output_path = 'pose_output/'
    if not os.path.isdir('pose_output'):
        os.makedirs('pose_output')

    # Load example scene
    scene_folder = './'
    Thread.thread().file_resolver().append(scene_folder)
    scene = xml.load_file('out_scene_for_pose.xml', integrator="flag")

    ref = read_exr('color_proj_dist.exr')

    params = traverse(scene)

    # Discard all parameters except for one we want to differentiate
    params.keep(["Projector.position", "Projector.rotation"])

    params.update()

    iter_lr = [(1000, 1),
               (2500, 0.08)]

    it = 0

    time_a = time.time()

    for (start_iter, lr) in iter_lr:

        opt = Adam(params, lr=lr)

        while True:

            # Perform a differentiable rendering of the scene
            with Float.Scope("rendering..."):
                image = render(scene, optimizer=opt, spp=1, unbiased=True)
            
            if it%20 == 0:
                write_bitmap(output_path + 'out_%07i.png' % it, image, (640, 480))
            
            with Float.Scope("computing loss..."):
                ob_val = ek.hsum(ek.sqr(image - ref)) / len(image)


            ek.backward(ob_val)

            # Optimizer: take a gradient step -> update displacement map
            opt.step()

            if it%20 == 0:
                print('[Enoki] Iteration %03i: error=%g' % (it, ob_val[0]), params["Projector.position"], params["Projector.rotation"])
            it += 1
                        
            if it == start_iter:
                break
                
    time_b = time.time()
    print('total : %f' % (((time_b - time_a) * 1000000)), 's')

    f = open("projector_pose.xml", 'w')
    f.write(projector_pose_template(params["Projector.position"], params["Projector.rotation"]))
    f.close()
