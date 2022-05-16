import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
import cv2
from cv2 import imwrite
import potpourri3d as pp3d
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from utils import *
from dmtet import *
import potpourri3d as pp3d
import polyscope as ps
from torch.nn import functional

integrator = psdr_cuda.DirectIntegrator()
scene = psdr_cuda.Scene()
scene.load_file('data/scenes/sphere_env_largesteps.xml', False)
scene.opts.spp = 32
scene.opts.width = 284
scene.opts.height = 216

# dmtet = DMTetGeometry(32, 2.1)
# v, f, uvs, uv_idx = dmtet.getMesh()
# scene.param_map['Mesh[0]'].load_mem(Vector3fD(v), Vector3iD(f.to(torch.int32)),
#                     Vector2fD(uvs), Vector3iD(uv_idx.to(torch.int32)), True)
scene.param_map['Mesh[0]'].load()
scene.configure()
img = integrator.renderC(scene, 2)
save_img(img, 'img1.png', (284, 216))

scene.param_map['Mesh[0]'].load('data/meshes/sphere_2k.obj')
scene.configure()
img = integrator.renderC(scene, 2)
save_img(img, 'img.png', (284, 216))
# for i, id in enumerate([2, 13]):
#     save_img(imgs[i], f'img.{id}.png', (284, 216))

# v, f = pp3d.read_mesh('mesh.obj')
# ps.init()
# ps.register_surface_mesh('m', v, f)
# ps.show()

'''
How to render a dmtet mesh
    - get v and f of the dmtet mesh and then overwrite the 
    corresponding properties of a mesh already loaded in param_map
    - Save dmtet as an obj and then load
'''
