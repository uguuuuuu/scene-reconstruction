import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from igl import hausdorff
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
import torch
from torch.optim import Adam
from torch.nn import functional
from torch.utils.data import DataLoader
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from util import *
from render.loss import get_loss_fn
from dataset import DatasetMesh
from geometry.util import laplacian_uniform, regularization_loss, remesh

FLAGS = json.load(open('configs/spot_env.json', 'r'))
batch_size = FLAGS['batch_size']
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
key = FLAGS['key']
res = FLAGS['res']
spp_ref, spp_opt = FLAGS['spp_ref'], FLAGS['spp_opt']
loss_fn = get_loss_fn(FLAGS['loss_fn'], FLAGS['tonemap'])
lambda_, alpha = FLAGS['lambda'], FLAGS['alpha']
if alpha == 0.: alpha = None
lr = FLAGS['learning_rate']
outdir = f"output/{FLAGS['name']}"
scene_info = preprocess_scene(f"{FLAGS['scene']}.xml")
n_sensors = scene_info['n_sensors']
sensor_ids = range(n_sensors)

for i in range(n_sensors):
    os.makedirs(f'{outdir}/train/{sensor_ids[i]}', exist_ok=True)
os.makedirs(f'{outdir}/stats', exist_ok=True)
os.makedirs(f'{outdir}/ref', exist_ok=True)
os.makedirs(f'{outdir}/optimized', exist_ok=True)

print('Rendering reference images...')
integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')
scene, ref_imgs = renderC_img(scene_info['tgt'], integrator, res=res, spp=spp_ref)
scene, ref_masks = renderC_img(scene_info['tgt'], integrator_mask, res=res, spp=spp_ref)
for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
    save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/refmask_{sensor_ids[i]}.exr', res)
ref_v, ref_f = scene.get_mesh(key, True)
print('Finished')

print('Rendering initial images...')
scene = Scene(scene_info['src'])
scene.reload_mat(key, load_img('data/meshes/texture_kd.exr'))
scene.set_opts(res, 32, sppe=0, sppse=0)
init_imgs = scene.renderC(integrator)
init_masks = scene.renderC(integrator_mask)
for i, (img, mask) in enumerate(zip(init_imgs, init_masks)):
    save_img(img, f'{outdir}/ref/init_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/initmask_{sensor_ids[i]}.exr', res)
del init_imgs, init_masks, scene
print("Finished")

ref_imgs = torch.stack([torch.from_numpy(img).cuda() for img in ref_imgs])
ref_masks = torch.stack([torch.from_numpy(mask).cuda() for mask in ref_masks])
trainset = DatasetMesh(sensor_ids, ref_imgs, ref_masks)
print('Loading checkpoint...')
if start_itr != 0:
    v, start_itr = load_ckp(f'{outdir}/optimized/vertex_positions.*.pt', start_itr)
    if v is not None:
        texture = load_img(f'{outdir}/optimized/texture_kd.{start_itr}.exr')
        print(f'Loaded checkpoint from epoch {start_itr}')
    else:
        texture = None
else:
    v = texture = None
print('Finished')
scene = Scene(scene_info['src'])
if v is None: 
    v, f = scene.get_mesh(key)
    v, f = torch.from_numpy(v).cuda(), torch.from_numpy(f).cuda()
else:
    _, f = scene.get_mesh(key)
    f = torch.from_numpy(f).cuda()
with torch.no_grad():
    M = compute_matrix(v, f, lambda_, alpha=alpha)
    u: torch.Tensor = to_differential(M, v)
    L = laplacian_uniform(v.shape[0], f)
u.requires_grad_()
if texture is None:
    texture = load_img('data/meshes/texture_kd.exr')
scene.reload_mat(key, texture)
scene.set_opts(res, spp=spp_opt, sppe=spp_opt, sppse=spp_opt)
texture = torch.from_numpy(texture).reshape(-1,3).cuda().contiguous()
texture.requires_grad_()
opt_geom = AdamUniform([u], lr)
opt_mat = Adam([texture], lr)
img_losses = []
reg_losses = []
mask_losses = []
distances = []

for it in tqdm(range(start_itr, start_itr + n_itr)):
    img_loss_, mask_loss_ = 0., 0.
    for ids, ref_imgs, ref_masks in DataLoader(trainset, batch_size, True):
        v = from_differential(M, u, method='CG')

        imgs = scene.renderDVAM(v, texture, key, integrator, integrator_mask, ids)

        img_loss = loss_fn(imgs[0], ref_imgs)
        mask_loss = functional.mse_loss(imgs[1], ref_masks)
        loss = img_loss + mask_loss

        opt_geom.zero_grad()
        opt_mat.zero_grad()
        loss.backward()
        opt_geom.step()
        opt_mat.step()

        with torch.no_grad():
            texture.clamp_(0., 1.)
            img_loss_ += img_loss
            mask_loss_ += mask_loss
            if it % 4 == 0:
                for i, id in enumerate(ids):
                    save_img(imgs[0][i], f'{outdir}/train/{id}/train.{it+1:04}.png', res)
                    save_img(imgs[1][i], f'{outdir}/train/{id}/train_mask.{it+1:04}.exr', res)

    with torch.no_grad():
        if (it+1) % save_interval == 0:
            torch.save(v.detach(), f'{outdir}/optimized/vertex_positions.{it+1}.pt')
            save_img(texture, f'{outdir}/optimized/texture_kd.{it+1}.exr', (1024, 1024))
        img_losses.append(img_loss_.cpu().numpy())
        mask_losses.append(mask_loss_.cpu().numpy())
        reg_losses.append(regularization_loss(L, v, True).cpu().numpy())
        v, f = scene.get_mesh(key, True)
        distances.append(hausdorff(v, f, ref_v, ref_f))
    # Remesh
    # if it == FLAGS['remesh_itr']:
    #     with torch.no_grad():
    #         v = from_differential(M, u, method='CG')
    #         _, f = scene.get_mesh(key)
    #         v, f = remesh(v, torch.from_numpy(f).cuda())
    #         scene.reload_mesh(key, v, f)
    #         scene.set_opts(res, spp=32, sppe=0, sppse=0)
    #         remesh_imgs = scene.renderC(integrator)
    #         remesh_masks = scene.renderC(integrator_mask)
    #         for i, id in enumerate(sensor_ids):
    #             save_img(remesh_imgs[i], f'{outdir}/ref/remesh_{id}.png', res)
    #             save_img(remesh_masks[i], f'{outdir}/ref/remeshmask_{id}.exr', res)
    #         scene.set_opts(res, spp=spp_opt, sppe=spp_opt, sppse=spp_opt)
    #         v, f = scene.get_mesh(key)
    #         v, f = torch.from_numpy(v).cuda(), torch.from_numpy(f).cuda()
    #         M = compute_matrix(v, f, lambda_, alpha=alpha)
    #         u: torch.Tensor = to_differential(M, v)
    #         L = laplacian_uniform(v.shape[0], f)
    #     u.requires_grad_()
    #     opt = AdamUniform([u, texture], lr)

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
    v = from_differential(M, u, method='CG')
    _, f, uvs, uv_idx = scene.get_mesh(key, return_uv=True)
    scene.reload_mesh(key, v, f, uvs, uv_idx)
    scene.reload_mat(key, texture.reshape(1024, 1024, -1))
    scene.set_opts(res, spp=32, sppe=0, sppse=0)
    imgs = scene.renderC(integrator)
    masks = scene.renderC(integrator_mask)
    for i in range(n_sensors):
        save_img(imgs[i], f'{outdir}/optimized/itr{it+1}_{i}.png', res)
        save_img(masks[i], f'{outdir}/optimized/itr{it+1}_{i}_mask.exr', res)
    scene.dump(key, f'{outdir}/optimized/optimized_itr{it+1}_l{lambda_}_lr{lr}.obj')
    save_img(texture, f'{outdir}/optimized/texture_kd_itr{it+1}_l{lambda_}_lr{lr}.exr', (1024, 1024))

for id in sensor_ids:
    imgs2video(f"{outdir}/train/{id}/train.*.png", f"{outdir}/train/{id}.mp4", 30)
    imgs2video(f"{outdir}/train/{id}/train_mask.*.exr", f"{outdir}/train/{id}_mask.mp4", 30)

'''
Issues:
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
    - Training images too noisy
        - Increase spp and res
    - The optimized textures are too noisy
'''

'''
Observations:
'''

'''
Notes:
    - Mask Loss
        - Consider depth mask
        - Obtain mask images for real-world input images (using Detectron2 for example)
'''

'''
    TODO
    - Add more scenes (with effects like indirect illumination, shadows, refractions, etc.)
        - Increase number of views
        - Try different materials
    - Use tonemapped colors in computing losses as in nvdiffrec instead of linear ones
    - Use scheduler during training
        - try to use exponential falloff
    - Try real-world images
    - Implement upsampling and downsampling remesh algorithms
    - *Jointly optimize shape and material*
        - try different shading models
        - try different coordinate networks
        - use mipmapped textures
        - remesh and upsample textures periodically during optimization
'''