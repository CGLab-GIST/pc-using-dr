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

# This script generates 2 mitsuba scenes from color and depth images.
# 1. Direct light scene for estimating pose and W
# 2. Indirect light scene (path tracing) for estimating optimal input images

import os

import open3d as o3d
import cv2
import pymeshlab

import socket
import time

import numpy as np
import imageio
from common import *

def scene_template(fov, width, height, mesh_filename, texture_filename):
    return \
        "<?xml version='1.0' encoding='utf-8'?>\n"+\
        "<scene version=\"0.5.0\">\n"+\
        "    <integrator type=\"path\"/>\n"+\
        "    <sensor type=\"perspective\">\n"+\
        "        <float name=\"fov\" value=\"" + str(fov) + "\"/>\n"+\
        "        <transform name=\"toWorld\">\n"+\
        "            <matrix value=\"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n"+\
        "        </transform>\n"+\
        "        <sampler type=\"independent\">\n"+\
        "            <integer name=\"sampleCount\" value=\"100\"/>\n"+\
        "        </sampler>\n"+\
        "        <film type=\"hdrfilm\">\n"+\
        "            <integer name=\"height\" value=\"" + str(height) + "\"/>\n"+\
        "            <integer name=\"width\" value=\"" + str(width) + "\"/>\n"+\
        "            <rfilter type=\"box\">\n"+\
        "                <float name=\"radius\" value=\"0.1\"/>\n"+\
        "            </rfilter>\n"+\
        "        </film>\n"+\
        "    </sensor>\n"+\
        "    <!-- Insert any emiiter -->\n"+\
        "    <emitter type=\"projector\">\n"+\
        "        <float name=\"fov\" value=\"25.76093166\"/>\n"+\
        "        <include filename=\"projector_pose.xml\"/>\n"+\
		"        <float name=\"shift_y\" value=\"0.65\"/>\n"+\
        "        <include filename=\"projector_scale.xml\"/>\n"+\
        "        <texture type=\"bitmap\" name=\"irradiance\">\n"+\
        "            <string name=\"filename\" value=\"prj_texture_black.png\"/>\n"+\
        "        </texture>\n"+\
        "    </emitter>\n"+\
        "    <shape type=\"obj\">\n"+\
        "        <string name=\"filename\" value=\"" + mesh_filename + "\"/>\n"+\
        "        <transform name=\"toWorld\">\n"+\
        "            <rotate y=\"1\" angle=\"180\"/>\n"+\
        "        </transform>\n"+\
        "        <bsdf type=\"diffuse\">\n"+\
        "            <texture name=\"reflectance\" type=\"bitmap\">\n"+\
        "                <string name=\"filename\" value=\"" + texture_filename + "\"/>\n"+\
        "            </texture>\n"+\
        "        </bsdf>\n"+\
        "    </shape>\n"+\
        "</scene>\n"


# Scene without texture
# This scene will be used for pose optimization and W
def pose_scene_template(fov, width, height, mesh_filename):
    return \
        "<?xml version='1.0' encoding='utf-8'?>\n"+\
        "<scene version=\"0.5.0\">\n"+\
        "    <integrator type=\"$integrator\"/>\n"+\
        "    <sensor type=\"perspective\">\n"+\
        "        <float name=\"fov\" value=\"" + str(fov) + "\"/>\n"+\
        "        <transform name=\"toWorld\">\n"+\
        "            <matrix value=\"1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\"/>\n"+\
        "        </transform>\n"+\
        "        <sampler type=\"independent\">\n"+\
        "            <integer name=\"sampleCount\" value=\"100\"/>\n"+\
        "        </sampler>\n"+\
        "        <film type=\"hdrfilm\">\n"+\
        "            <integer name=\"height\" value=\"" + str(height) + "\"/>\n"+\
        "            <integer name=\"width\" value=\"" + str(width) + "\"/>\n"+\
        "            <rfilter type=\"box\">\n"+\
        "                <float name=\"radius\" value=\"0.1\"/>\n"+\
        "            </rfilter>\n"+\
        "        </film>\n"+\
        "    </sensor>\n"+\
        "    <!-- Insert any emiiter -->\n"+\
        "    <emitter type=\"projector\">\n"+\
        "        <float name=\"fov\" value=\"25.76093166\"/>\n"+\
        "        <include filename=\"projector_pose.xml\"/>\n"+\
		"        <float name=\"shift_y\" value=\"0.65\"/>\n"+\
        "        <float name=\"blur_size\" value=\"0.01\"/>\n"+\
        "        <float name=\"scale\" value=\"8000\"/>\n"+\
        "        <texture type=\"bitmap\" name=\"irradiance\">\n"+\
        "            <string name=\"filename\" value=\"prj_texture_white.png\"/>\n"+\
        "        </texture>\n"+\
        "    </emitter>\n"+\
        "    <shape type=\"obj\">\n"+\
        "        <string name=\"filename\" value=\"" + mesh_filename + "\"/>\n"+\
        "        <transform name=\"toWorld\">\n"+\
        "            <rotate y=\"1\" angle=\"180\"/>\n"+\
        "        </transform>\n"+\
        "    </shape>\n"+\
        "</scene>\n"


def projector_pose_template():
    return \
        "<?xml version='1.0' encoding='utf-8'?>\n"+\
        "<scene version=\"0.5.0\">\n"+\
        "    <point name=\"position\" value=\"0, 0, 0\"/>\n"+\
		"    <vector name=\"rotation\" value=\"0, 0, 0\"/>\n"+\
        "</scene>\n"

def scale_str(scale):
    return '''
<?xml version='1.0' encoding='utf-8'?>
<scene version="0.5.0">
    <float name="scale" value="''' + str(scale) + '''"/>
</scene>
    '''

if __name__ == '__main__':


    width = 640
    height = 480
    fx = 531.7256
    fy = 531.7256
    cx = 320
    cy = 240
    fov = 62.08

    os.chdir(scene_path)

    # Input
    color_filename = "color.png"
    depth_filename = "depth.exr"

    # Output
    scene_path = "."
    
    mesh_filename = "out_mesh.obj"
    texture_filename = "out_texture.png"
    scene_filename = "out_scene.xml"
    pose_scene_filename = "out_scene_for_pose.xml"
    
    # Cleanup
    if os.path.exists(mesh_filename):
        os.remove(mesh_filename)
    if os.path.exists(texture_filename):
        os.remove(texture_filename)
    if os.path.exists(scene_filename):
        os.remove(scene_filename)


    # Using open3d, convert RGBD image to point cloud
    # Read depth and color images
    print("Read depth and color images...")
    depth_exr = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
    depth_exr = cv2.resize(depth_exr, (width, height))
    color_png = cv2.imread(color_filename, cv2.IMREAD_UNCHANGED)
    color_png = cv2.cvtColor(color_png, cv2.COLOR_RGB2BGR)

    depth_img = o3d.geometry.Image(depth_exr)
    color_img = o3d.geometry.Image(color_png)
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_img, depth_img, convert_rgb_to_intensity=False)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # Generate point cloud
    print("Creating point cloud from rgbd...")
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.scale(100000, [[0],[0],[0]])

    pcd.estimate_normals()
    pcd_normal = np.asarray(pcd.normals)
    print(pcd_normal)
    pcd.normals = o3d.utility.Vector3dVector(pcd_normal)

    o3d.io.write_point_cloud("out_pointcloud.ply", pcd)

    # # Using MeshLab, convert point cloud to .obj and texture
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("out_pointcloud.ply")

    ms.compute_normals_for_point_sets(k=20, flipflag=True, viewpos=np.zeros(3))
    ms.surface_reconstruction_ball_pivoting()
    ms.parametrization_trivial_per_triangle(textdim=4096, border=0)
    ms.simplification_quadric_edge_collapse_decimation_with_texture(targetfacenum=int(ms.current_mesh().face_number()),
                                                                    preserveboundary=True,
                                                                    preservenormal=True,
                                                                    planarquadric=True)
    ms.invert_faces_orientation(forceflip=True)
    ms.normalize_vertex_normals()
    ms.normalize_face_normals()

    ms.close_holes(maxholesize = 40)
    ms.hc_laplacian_smooth()

    ms.transfer_vertex_color_to_texture(textname=texture_filename, textw=4096, texth=4096)
    ms.save_current_mesh(mesh_filename, save_vertex_color=False)

    # if os.path.exists("out_pointcloud.ply"):
    #     os.remove("out_pointcloud.ply")
    if os.path.exists(mesh_filename+".mtl"):
        os.remove(mesh_filename+".mtl")

    f = open(scene_filename, 'w')
    f.write(scene_template(fov, width, height, mesh_filename, texture_filename))
    f.close()

    f = open(pose_scene_filename, 'w')
    f.write(pose_scene_template(fov, width, height, mesh_filename))
    f.close()

    f = open("projector_pose.xml", 'w')
    f.write(projector_pose_template())
    f.close()

    f = open("projector_scale.xml", 'w')
    f.write(scale_str(50000))
    f.close()

    print("You must check directions of normal vectors...")
    print("If flipped, comment line 189 :")
    print("    \'ms.invert_faces_orientation(forceflip=True)\'")
