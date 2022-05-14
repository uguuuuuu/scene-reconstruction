import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from igl import hausdorff
import potpourri3d as pp3d
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
import torch
from torch.optim import Adam
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from utils import *

'''
Set up parameters
'''
target_name = 'bunny_env_largesteps'
init_name = 'sphere_env_largesteps'
n_sensors = 2
# sensor_ids = [0, 2, 7, 9, 12, 14]
# sensor_ids = [2, 3, 12, 13]
sensor_ids = [2, 13]
key = 'Mesh[0]'
res = (284, 216)
use_adam = False
use_uniform = True # Use Largesteps' optimization method
use_bilaplacian = False
loss_fn = 'l1'
start_itr = None # start from the latest saved tensor
n_itr = 1000
save_interval = 500
lr = 1e-1
lambda_ = 99 # Hyperparameter of the reparametrization matrix
alpha = None # Hyperparameter of the reparametrization matrix
l = 0. # Hyperparameter of the regularization loss
if use_adam:
    outdir = f'output/{target_name}/adam/sensors_{n_sensors}'
elif l != 0.:
    outdir = f'output/{target_name}/regularized/sensors_{n_sensors}'
elif use_uniform:
    outdir = f'output/{target_name}/uniform_adam/sensors_{n_sensors}'
else:
    raise NotImplementedError
for i in range(n_sensors):
    os.makedirs(f'{outdir}/{sensor_ids[i]}', exist_ok=True)
if start_itr != 0:
    v, start_itr = load_tensor(f'{outdir}/vertex_positions.*.pt', start_itr)
    if v is not None:
        assert(str(v.device) == 'cuda:0')
        assert(v.requires_grad == False)
else:
    v = None

'''
Record initial state 
'''
integrator = psdr_cuda.DirectIntegrator(bsdf_samples=2, light_samples=2)
scene, ref_imgs = renderC_img(f'data/scenes/{target_name}.xml', integrator, sensor_ids, res)
for i, ref in enumerate(ref_imgs):
    save_img(ref, f'{outdir}/ref_{sensor_ids[i]}.png', scene)
    ref_imgs[i] = ref.torch()
ref_imgs = torch.stack(ref_imgs)
ref_v = scene.param_map[key].vertex_positions.numpy()
ref_f = scene.param_map[key].face_indices.numpy()

scene, init_imgs = renderC_img(f'data/scenes/{init_name}.xml', integrator, sensor_ids, res)
for i, init in enumerate(init_imgs):
    save_img(init, f'{outdir}/init_{sensor_ids[i]}.png', scene)
del init_imgs

'''
Prepare for optimization
'''
with torch.no_grad():
    if v is None: v = scene.param_map[key].vertex_positions.torch()
    assert(v.requires_grad == False)
    M = compute_matrix(v, scene.param_map[key].face_indices.torch(), lambda_, alpha=alpha)
    assert(M.requires_grad == False)
    u: torch.Tensor = to_differential(M, v)
    laplacian = laplacian_uniform(v.shape[0], scene.param_map[key].face_indices)
    assert(laplacian.requires_grad == False)
    loss_fn = get_loss_fn(loss_fn)

u.requires_grad_()
opt = AdamUniform([u], lr)
scene.opts.spp = 1
scene.configure()

img_losses = []
reg_losses = []
distances = []
'''
Training Loop
'''
for it in tqdm(range(start_itr, start_itr + n_itr)):
    v = from_differential(M, u, method='CG')
    imgs = renderD({
            'scene': scene,
            'key': key,
            'integrator': integrator,
            'sensor_ids': sensor_ids
        }, v)
    for i, id in enumerate(sensor_ids):
        save_img(imgs[i], f'{outdir}/{id}/train.{it+1:03}.png', scene)

    img_loss = loss_fn(imgs, ref_imgs)
    loss = img_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        img_losses.append(img_loss.detach().cpu().numpy())
        reg_losses.append(regularization_loss(laplacian, v, True).cpu().numpy())
        distances.append(hausdorff(v.detach().cpu().numpy(), scene.param_map[key].face_indices.numpy(),
                    ref_v, ref_f))
        if (it+1) % save_interval == 0:
            torch.save(v.detach(), f'{outdir}/vertex_positions.{it+1}.pt')

'''
Save results
'''
plt.plot(img_losses, label='Image Loss')
plt.plot(reg_losses, label='Regularization Loss')
plt.yscale('log')
plt.legend()
plt.savefig(f'{outdir}/losses_itr{it+1}_l{lambda_}_lr{lr}.png')
plt.close()
plt.plot(distances)
plt.ylabel("Hausdorff Distance")
plt.savefig(f'{outdir}/distances_itr{it+1}_l{lambda_}_lr{lr}.png')

with torch.no_grad():
    scene.opts.spp = 32
    v = from_differential(M, u, method='CG')
    scene.param_map[key].vertex_positions = Vector3fD(v)
    scene.configure()
    for i in sensor_ids:
        save_img(integrator.renderC(scene, sensor_id=i), f'{outdir}/itr{it+1}_{i}.png', scene)
    v = scene.param_map[key].vertex_positions.numpy()
    f = scene.param_map[key].face_indices.numpy()
    pp3d.write_mesh(v, f, f'{outdir}/optimized_itr{it+1}_l{lambda_}_lr{lr}.obj')

'''
Issues:
    - When the material is highly specular, maybe need to add mask loss (since the surface
    is similar to the background in terms of the pixel values)
'''

'''
Observations:
    - Largesteps' method makes the mesh smooth
    - lambda_ controls how smooth the mesh is during optimization
'''

'''
    TODO
    - Add more scenes
    - Use gamma-corrected RGB colors in computing losses instead of linear ones
    - *Implement DMTet*
    - *Use DMTet to optimize a torus to a bunny*
    - Add mask loss
    - *Jointly optimize shape and material (fit a diffuse bsdf first)*
        - implement or use a uv mapping algorithm (e.g. BFF, xatlas)
'''