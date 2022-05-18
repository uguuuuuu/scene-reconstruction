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

# scene_xmls = preprocess_scene('data/scenes/spot_env/spot_env.xml')
sensor_ids = range(16)
res = (284, 216)
integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')
scene, ref_imgs = renderC_img('data/scenes/spot_env/spot_env.xml', integrator, sensor_ids, res, load_string=False)
# _, ref_masks = renderC_img(scene_xmls['tgt_mask'], integrator_mask, sensor_ids, res)
# for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
#     ref_imgs[i] = img.torch()
#     ref_masks[i] = mask.torch()
# ref_imgs = torch.stack(ref_imgs)
# ref_masks = torch.stack(ref_masks)

v = scene.param_map['Mesh[0]'].vertex_positions.torch()
f = scene.param_map['Mesh[0]'].face_indices.torch()
m = remesh(v, f)
obj.write_obj('mesh.obj', m)

ps.init()

ps.register_surface_mesh('original', v.cpu().numpy(), f.cpu().numpy())
ps.register_surface_mesh('remeshed', m.v_pos.cpu().numpy(), m.t_pos_idx.cpu().numpy())

ps.show()