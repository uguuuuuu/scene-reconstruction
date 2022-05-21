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

integrator = psdr_cuda.DirectIntegrator(1, 1)
res = (320, 180)
scene_info = preprocess_scene('example.xml')
scene = Scene(scene_info['tgt'])
scene.set_opts(res, 128, sppe=0, sppse=0)
scene.prepare()
for _ in range(100):
    img = integrator.renderC(scene._scene, 2)
save_img(img, 'img.exr', res)