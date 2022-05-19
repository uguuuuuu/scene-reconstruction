import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
sys.path.append('ext/large-steps-pytorch/ext/botsch-kobbelt-remesher-libigl/build')

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

v = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])
mat = Matrix4fD.translate(Vector3fD(2., 3., 4.)).numpy().squeeze()
print(v)
v = transform_np(v, mat)
print(v)