import os
import random
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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

for i in range(1, 16):
    imgs2video(f'output/bunny_env_largesteps/uniform_adam/sensors_16/train/{i}/train.*.png',
            f'output/bunny_env_largesteps/uniform_adam/sensors_16/train/{i}/{i}.mp4', 30)
    imgs2video(f'output/bunny_env_largesteps/uniform_adam/sensors_16/train/{i}/train_mask.*.exr',
            f'output/bunny_env_largesteps/uniform_adam/sensors_16/train/{i}/{i}_mask.mp4', 30)