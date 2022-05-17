import os
import random
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
from renderD import renderDVA
import config

'''
Set up parameters
'''
FLAGS = json.load(open('configs/bunny_env_largesteps.json', 'r'))
n_sensors, batch_size = FLAGS['n_sensors'], FLAGS['batch_size']
sensor_ids = FLAGS['sensor_ids'] if batch_size == 0 else range(n_sensors)
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
    os.makedirs(f'{outdir}/train/{sensor_ids[i]}', exist_ok=True)
os.makedirs(f'{outdir}/stats', exist_ok=True)
os.makedirs(f'{outdir}/ref', exist_ok=True)
os.makedirs(f'{outdir}/optimized', exist_ok=True)
if start_itr != 0:
    v, start_itr = load_tensor(f'{outdir}/optimized/vertex_positions.*.pt', start_itr)
    if v is not None:
        assert(str(v.device) == 'cuda:0')
        assert(v.requires_grad == False)
else:
    v = None

'''
Record initial state 
'''
integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')
scene, ref_imgs = renderC_img(f"{FLAGS['target_scene']}.xml", integrator, sensor_ids, res)
_, ref_masks = renderC_img(f"{FLAGS['target_scene']}_mask.xml", integrator_mask, sensor_ids, res)
for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
    save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/refmask_{sensor_ids[i]}.exr', res)
    ref_imgs[i] = img.torch()
    ref_masks[i] = mask.torch()
ref_imgs = torch.stack(ref_imgs)
ref_masks = torch.stack(ref_masks)
to_world = scene.param_map[key].to_world.torch().squeeze()
ref_v = transform(scene.param_map[key].vertex_positions.torch(), to_world).cpu().numpy()
ref_f = scene.param_map[key].face_indices.numpy()

scene, init_imgs = renderC_img(f"{FLAGS['source_scene']}.xml", integrator, sensor_ids, res)
scene_mask, init_masks = renderC_img(f"{FLAGS['source_scene']}_mask.xml", integrator_mask, sensor_ids, res)
for i, (img, mask) in enumerate(zip(init_imgs, init_masks)):
    save_img(img, f'{outdir}/ref/init_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/initmask_{sensor_ids[i]}.exr', res)
del init_imgs, init_masks
to_world = scene.param_map[key].to_world.torch().squeeze()

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
scene_mask.opts.spp = 1
scene_mask.configure()
config.scene = scene
config.scene_mask = scene_mask
config.integrator = integrator
config.integrator_mask = integrator_mask
config.key = key

img_losses = []
reg_losses = []
mask_losses = []
distances = []
'''
Training Loop
'''
for it in tqdm(range(start_itr, start_itr + n_itr)):
    ids = random.sample(sensor_ids, batch_size)
    config.sensor_ids = sensor_ids if batch_size == 0 else ids
    v = from_differential(M, u, method='CG')
    imgs = renderDVA(v)
    if it % 2 == 0:
        for i, id in enumerate(ids):
            save_img(imgs[0][i], f'{outdir}/train/{id}/train.{it+1:04}.png', res)
            save_img(imgs[1][i], f'{outdir}/train/{id}/train_mask.{it+1:04}.exr', res)

    img_loss = loss_fn(imgs[0], ref_imgs[ids])
    mask_loss = functional.mse_loss(imgs[1], ref_masks[ids])
    loss = img_loss + mask_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        if (it+1) % save_interval == 0:
            torch.save(v.detach(), f'{outdir}/optimized/vertex_positions.{it+1}.pt')
        img_losses.append(img_loss.detach().cpu().numpy())
        mask_losses.append(mask_loss.detach().cpu().numpy())
        reg_losses.append(regularization_loss(laplacian, v, True).cpu().numpy())
        v = transform(v, to_world)
        distances.append(hausdorff(v.detach().cpu().numpy(), scene.param_map[key].face_indices.numpy(),
                    ref_v, ref_f))

'''
Save results
'''
plt.plot(img_losses, label='Image Loss')
plt.plot(mask_losses, label='Mask Loss')
plt.plot(reg_losses, label='Regularization Loss')
plt.yscale('log')
plt.legend()
plt.savefig(f'{outdir}/stats/losses_itr{it+1}_l{lambda_}_lr{lr}.png')
plt.close()
plt.plot(distances)
plt.ylabel("Hausdorff Distance")
plt.savefig(f'{outdir}/stats/distances_itr{it+1}_l{lambda_}_lr{lr}.png')

with torch.no_grad():
    scene.opts.spp = 32
    v = from_differential(M, u, method='CG')
    scene.param_map[key].vertex_positions = Vector3fD(v)
    scene.configure()
    for i in sensor_ids:
        save_img(integrator.renderC(scene, i), f'{outdir}/optimized/itr{it+1}_{i}.png', res)
        save_img(integrator_mask.renderC(scene_mask, i), f'{outdir}/optimized/itr{it+1}_{i}_mask.exr', res)
    scene.param_map[key].dump(f'{outdir}/optimized/optimized_itr{it+1}_l{lambda_}_lr{lr}.obj')

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
        - Coverage mask or depth mask?
        - How many channels, one or three? (maybe less memory consumption with one channel)
        - Obtain mask images for real-world input images (using Detectron2 for example)
    - Use scheduler during training
        - try to use exponential falloff
    - Implement a Dataset
    - *Implement the two-stage pipeline to optimize geometry*
    - *Jointly optimize shape and material (fit a diffuse bsdf first)*
        - implement or use a uv mapping algorithm (e.g. BFF, xatlas)
'''