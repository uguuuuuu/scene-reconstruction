import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys

import json
import cv2
import potpourri3d as pp3d
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from util import *
from geometry.dmtet import *
from geometry import obj, mesh, util
from dataset import DatasetMesh
from torch.utils.data import DataLoader
import potpourri3d as pp3d
import polyscope as ps
from torch.nn import functional
import numpy as np
import xml.etree.ElementTree as ET
from GPUtil import showUtilization
import igl

def test_mem_capacity(res, spp):
    integrator = psdr_cuda.DirectIntegrator(1, 1)
    scene_info = preprocess_scene('example.xml')
    scene = Scene(scene_info['tgt'])
    scene.set_opts(res, spp, sppe=0, sppse=0)
    scene.prepare()
    for _ in range(100):
        img = integrator.renderC(scene._scene, 2)
    save_img(img, 'img.exr', res)

def display_meshes(fnames):
    ps.init()
    for i, fname in enumerate(fnames):
        v, f = pp3d.read_mesh(fname)
        ps.register_surface_mesh(f'm{i}', v + np.array([5, 0, 0]) * i, f, False)
    ps.show()

display_meshes([
    'data/meshes/bunny/bunny.obj',
    *glob('output/bunny_env_largesteps_dmtet/optimized/optimized*.obj')
])