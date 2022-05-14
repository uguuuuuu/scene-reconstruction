from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from utils import *
from dmtet import *

integrator = psdr_cuda.DirectIntegrator()
scene = psdr_cuda.Scene()
scene.load_file('data/scenes/sphere_env_largesteps.xml', False)
scene.opts.spp = 1
scene.opts.width = 284
scene.opts.height = 216
print(scene.param_map['Mesh[0]'])
help(scene.param_map['Mesh[0]'])

# dmtet = DMTetGeometry(64, 2.1)
# imgs = dmtet(scene, 'Mesh[0]', integrator, [2, 13])
# for i, id in enumerate([2, 13]):
#     save_img(imgs[i], f'img.{id}.png', scene)