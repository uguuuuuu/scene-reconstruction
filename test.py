import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys

import json
import cv2
import potpourri3d as pp3d
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from utils import *
from geometry.dmtet import *
from geometry import obj, mesh, util
from dataset import DatasetMesh
from torch.utils.data import DataLoader
import potpourri3d as pp3d
import polyscope as ps
from torch.nn import functional
import numpy as np
import xml.etree.ElementTree as ET

res = (320, 180)
integrator = psdr_cuda.DirectIntegrator(1, 1)
scene = psdr_cuda.Scene()
scene.load_file('example.xml', False)
scene.opts.width = res[0]
scene.opts.height = res[1]
scene.opts.spp = 32
# scene.configure()

# img = integrator.renderC(scene, 2)
# save_img(img, 'img.png', res)

dmtet = DMTetGeometry(32, 4)
tet_mesh = dmtet.getMesh()
v = Vector3fD(tet_mesh.v_pos)
f = Vector3iD(tet_mesh.t_pos_idx.to(torch.int32))
# uv = Vector2fD(tet_mesh.v_tex)
# uv_idx = Vector3iD(tet_mesh.t_tex_idx.to(torch.int32))
print(tet_mesh.v_pos.shape)
print(tet_mesh.t_pos_idx.shape)
print(tet_mesh.v_tex.shape)
print(tet_mesh.t_tex_idx.shape)
# m = scene.param_map['Mesh[0]']
# scene.reload_mesh_mem(m, v, f, uv, uv_idx)
# scene.configure()

# img = integrator.renderC(scene, 2)
# save_img(img, 'img1.png', res)