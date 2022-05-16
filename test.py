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
from geometry.dmtet import *
from geometry import obj, mesh, util
import potpourri3d as pp3d
import polyscope as ps
from torch.nn import functional

integrator = psdr_cuda.DirectIntegrator()
scene = psdr_cuda.Scene()
scene.load_file('data/scenes/spot_env.xml', False)
scene.opts.spp = 32
scene.opts.width = 284
scene.opts.height = 216

m = scene.param_map['Mesh[0]']
v = m.vertex_positions.torch()
f = m.face_indices.torch()
uvs = m.vertex_uv.torch()
uv_idx = m.face_uv_indices.torch()
uvs[:,1] = 1. - uvs[:,1]
m = mesh.Mesh(v, f, v_tex=uvs, t_tex_idx=uv_idx)
m = mesh.auto_normals(m)
obj.write_obj('.', m, True)


'''
How to render a dmtet mesh
    - get v and f of the dmtet mesh and then overwrite the 
    corresponding properties of a mesh already loaded in param_map
    - Save dmtet as an obj and then load
'''
