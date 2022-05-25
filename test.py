import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys

import json
import cv2
import potpourri3d as pp3d
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from torch.utils.data import DataLoader
import potpourri3d as pp3d
import polyscope as ps
from torch.nn import functional
import numpy as np
import xml.etree.ElementTree as ET
from GPUtil import showUtilization
import igl
import xatlas
import matplotlib.pyplot as plt

from util import *
from geometry.dmtet import *
from render.util import gen_tex, sample, wrap

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
        ps.register_surface_mesh(f'm{i}', v, f, False)
    ps.show()

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.cat([a,b], dim=0))