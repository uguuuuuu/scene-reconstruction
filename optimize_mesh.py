import os
import random
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from igl import hausdorff
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from utils import *
from loss import get_loss_fn
from renderD import renderDVA
from dataset import DatasetMesh
import config
from geometry import obj

'''
Set up parameters
'''
FLAGS = json.load(open('configs/bunny_env_largesteps.json', 'r'))
n_sensors, batch_size = FLAGS['n_sensors'], FLAGS['batch_size']
sensor_ids = range(n_sensors)
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
key = FLAGS['key']
res = FLAGS['res']
loss_fn = get_loss_fn(FLAGS['loss_fn'], FLAGS['tonemap'])
lambda_, alpha = FLAGS['lambda'], FLAGS['alpha']
if alpha == 0.: alpha = None
lr = FLAGS['learning_rate']
outdir = f"output/{FLAGS['name']}"
REMESH_PATH = f'{outdir}/optimized/remeshed.obj'
scene_xmls = preprocess_scene(f"{FLAGS['scene']}.xml", REMESH_PATH)

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
scene, ref_imgs = renderC_img(scene_xmls['tgt'], integrator, sensor_ids, res)
_, ref_masks = renderC_img(scene_xmls['tgt_mask'], integrator_mask, sensor_ids, res)
for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
    save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/refmask_{sensor_ids[i]}.exr', res)
ref_imgs = torch.stack([img.torch() for img in ref_imgs])
ref_masks = torch.stack([mask.torch() for mask in ref_masks])
trainset = DatasetMesh(sensor_ids, ref_imgs, ref_masks)
to_world = scene.param_map[key].to_world.torch().squeeze()
ref_v = transform(scene.param_map[key].vertex_positions.torch(), to_world).cpu().numpy()
ref_f = scene.param_map[key].face_indices.numpy()

scene, init_imgs = renderC_img(scene_xmls['src'], integrator, sensor_ids, res)
scene_mask, init_masks = renderC_img(scene_xmls['src_mask'], integrator_mask, sensor_ids, res)
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
    laplacian = laplacian_uniform(v.shape[0], scene.param_map[key].face_indices.torch())
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
    img_loss_, mask_loss_ = 0., 0.
    for ids, ref_imgs, ref_masks in DataLoader(trainset, batch_size, True):
        v = from_differential(M, u, method='CG')
        config.sensor_ids = ids

        imgs = renderDVA(v)

        img_loss = loss_fn(imgs[0], ref_imgs)
        mask_loss = functional.mse_loss(imgs[1], ref_masks)
        loss = img_loss + mask_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            img_loss_ += img_loss
            mask_loss_ += mask_loss
            if it % 4 == 0:
                for i, id in enumerate(ids):
                    save_img(imgs[0][i], f'{outdir}/train/{id}/train.{it+1:04}.png', res)
                    save_img(imgs[1][i], f'{outdir}/train/{id}/train_mask.{it+1:04}.exr', res)

    with torch.no_grad():
        if (it+1) % save_interval == 0:
            torch.save(v.detach(), f'{outdir}/optimized/vertex_positions.{it+1}.pt')
        img_losses.append(img_loss_.cpu().numpy())
        mask_losses.append(mask_loss_.cpu().numpy())
        reg_losses.append(regularization_loss(laplacian, v, True).cpu().numpy())
        distances.append(hausdorff(transform(v, to_world).cpu().numpy(),
                scene.param_map[key].face_indices.numpy(), ref_v, ref_f))
    # Remesh
    if it == FLAGS['remesh_itr']:
        with torch.no_grad():
            v = from_differential(M, u, method='CG')
            obj.write_obj(REMESH_PATH, remesh(v, scene.param_map[key].face_indices.torch()))
            scene, remesh_imgs = renderC_img(scene_xmls['rm'], integrator, sensor_ids, res)
            scene_mask, remesh_masks = renderC_img(scene_xmls['rm_mask'], integrator_mask, sensor_ids, res)
            for i, id in enumerate(sensor_ids):
                save_img(remesh_imgs[i], f'{outdir}/ref/remesh_{id}.png', res)
                save_img(remesh_masks[i], f'{outdir}/ref/remeshmask_{id}.exr', res)
            remesh_imgs = torch.stack([img.torch() for img in remesh_imgs])
            remesh_masks = torch.stack([mask.torch() for mask in remesh_masks])
            trainset = DatasetMesh(sensor_ids, remesh_imgs, remesh_masks)
            scene.opts.spp = 1
            scene_mask.opts.spp = 1
            config.scene = scene
            config.scene_mask = scene_mask
            lambda_ = 99
            lr /= 5
            v = scene.param_map[key].vertex_positions.torch()
            M = compute_matrix(v, scene.param_map[key].face_indices.torch(), lambda_, alpha=alpha)
            u: torch.Tensor = to_differential(M, v)
            laplacian = laplacian_uniform(v.shape[0], scene.param_map[key].face_indices.torch())
        u.requires_grad_()
        opt = AdamUniform([u], lr)

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

for id in sensor_ids:
    imgs2video(f"{outdir}/train/{id}/train.*.png", f"{outdir}/train/{id}.mp4", 30)
    imgs2video(f"{outdir}/train/{id}/train_mask.*.exr", f"{outdir}/train/{id}_mask.mp4", 30)

'''
Issues:
    - When the material is highly specular, maybe need to add mask loss (since the surface
    is similar to the background in terms of the pixel values)
    - The optimized mesh is smooth but lacks geometric details
        - Lower lambda_ (19 works well for the initial mesh)
        - Remesh periodically (also configure hyperparamters accordingly)
            - Try start remeshing at iteration 250 (instead of 500)
        - *Increase resolution and spp*
    - Remeshing results in the following optimization causing entangled geometry
        - Increase lambda_ after remeshing
        - Decrease the learning rate after remeshing
        - Increase spp
    - Remeshing loses UVs (causing problems when jointly optimizing geometry and materials)
        - Look up instant meshes
'''

'''
Observations:
    - Largesteps' method makes the mesh smooth
    - lambda_ controls how smooth the mesh is during optimization
    - lowering lambda_ from 99 to 19 significantly increased geometric details (
    compare the results after 1,000 iterations)
'''

'''
    TODO
    - Add more scenes (with effects like indirect illumination, shadows, refractions, etc.)
        - Increase number of views
        - Try different materials
        - Increase resolution and spp
    - Use tonemapped colors in computing losses as in nvdiffrec instead of linear ones
    - Add mask loss
        - Coverage mask or depth mask?
        - How many channels, one or three? (maybe less memory consumption with one channel?)
        - Obtain mask images for real-world input images (using Detectron2 for example)
    - Use scheduler during training
        - try to use exponential falloff
    - Try real-world images
    - *Implement upsampling and downsampling remesh algorithms*
    - *Implement the two-stage pipeline to optimize geometry*
    - *Jointly optimize shape and material (fit a diffuse bsdf first)*
        - implement or use a uv mapping algorithm (e.g. BFF, xatlas)
        - remesh and upsample textures periodically during optimization
'''