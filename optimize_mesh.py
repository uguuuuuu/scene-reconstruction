import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from igl import hausdorff
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
import config

'''
Set up parameters
'''
FLAGS = json.load(open('configs/bunny_env_largesteps.json', 'r'))
n_sensors, sensor_ids = FLAGS['n_sensors'], FLAGS['sensor_ids']
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
key = FLAGS['key']
res = FLAGS['res']
loss_fn = get_loss_fn(FLAGS['loss_fn'])
lambda_, alpha = FLAGS['lambda'], FLAGS['alpha']
if alpha == 0.: alpha = None
lr = FLAGS['learning_rate']
l = FLAGS['l']
if FLAGS['optimizer'] == 'adam':
    outdir = f"output/{FLAGS['name']}/adam/sensors_{FLAGS['n_sensors']}"
elif FLAGS['l'] > 0.:
    outdir = f"output/{FLAGS['name']}/regularized/sensors_{FLAGS['n_sensors']}"
elif FLAGS['optimizer'] == 'uniform':
    outdir = f"output/{FLAGS['name']}/uniform_adam/sensors_{FLAGS['n_sensors']}"
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
integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
scene, ref_imgs = renderC_img(FLAGS['target_scene'], integrator, sensor_ids, res)
for i, ref in enumerate(ref_imgs):
    save_img(ref, f'{outdir}/ref_{sensor_ids[i]}.png', res)
    ref_imgs[i] = ref.torch()
ref_imgs = torch.stack(ref_imgs)
ref_v = scene.param_map[key].vertex_positions.numpy()
ref_f = scene.param_map[key].face_indices.numpy()

scene, init_imgs = renderC_img(FLAGS['source_scene'], integrator, sensor_ids, res)
for i, init in enumerate(init_imgs):
    save_img(init, f'{outdir}/init_{sensor_ids[i]}.png', res)
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

u.requires_grad_()
opt = AdamUniform([u], lr)
scene.opts.spp = 1
scene.configure()
config.scene = scene
config.key = key
config.integrator = integrator
config.sensor_ids = sensor_ids

img_losses = []
reg_losses = []
distances = []
'''
Training Loop
'''
for it in tqdm(range(start_itr, start_itr + n_itr)):
    v = from_differential(M, u, method='CG')
    imgs = renderDV(v)
    for i, id in enumerate(sensor_ids):
        save_img(imgs[i], f'{outdir}/{id}/train.{it+1:03}.png', res)

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
        save_img(integrator.renderC(scene, sensor_id=i), f'{outdir}/itr{it+1}_{i}.png', res)
    scene.param_map[key].dump(f'{outdir}/optimized_itr{it+1}_l{lambda_}_lr{lr}.obj')

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
    - Add more scenes (with effects like indirect illumination, shadows, refractions, etc.)
    - Use gamma-corrected RGB colors in computing losses instead of linear ones
    - Add mask loss
        - Render mask images for synthetic scenes
        - Obtain mask images for real-world input images (using Detectron2 for example)
    - Use scheduler during training
    - *Implement the two-stage pipeline to optimize geometry*
    - *Jointly optimize shape and material (fit a diffuse bsdf first)*
        - implement or use a uv mapping algorithm (e.g. BFF, xatlas)
'''