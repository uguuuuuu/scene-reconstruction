import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from igl import hausdorff
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector2f as Vector2fD, Vector3f as Vector3fD, Vector3i as Vector3iD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
import torch
from torch.optim import Adam
from torch.nn import functional
from torch.utils.data import DataLoader
from utils import *
import config
from geometry.dmtet import DMTetGeometry, sdf_reg_loss
from geometry import obj
from dataset import DatasetMesh
from loss import get_loss_fn
from renderD import renderDVA

FLAGS = json.load(open('configs/bunny_env_largesteps_dmtet.json', 'r'))
n_sensors, batch_size = FLAGS['n_sensors'], FLAGS['batch_size']
sensor_ids = range(n_sensors)
key = FLAGS['key']
res = FLAGS['res']
spp_ref, spp_opt = FLAGS['spp_ref'], FLAGS['spp_opt']
loss_fn = get_loss_fn(FLAGS['loss_fn'], FLAGS['tonemap'])
sdf_weight = FLAGS['sdf_weight']
lr = FLAGS['learning_rate']
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
outdir = f"output/{FLAGS['name']}"
SDF_PATH = f'{outdir}/optimized/sdf.obj'
scene_xmls = preprocess_scene(f"{FLAGS['scene']}.xml", sdf_path=SDF_PATH)

for i in range(n_sensors):
    os.makedirs(f'{outdir}/train/{sensor_ids[i]}', exist_ok=True)
os.makedirs(f'{outdir}/stats', exist_ok=True)
os.makedirs(f'{outdir}/ref', exist_ok=True)
os.makedirs(f'{outdir}/optimized', exist_ok=True)

integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')
scene, ref_imgs = renderC_img(scene_xmls['tgt'], integrator, sensor_ids, res, spp_ref)
scene, ref_masks = renderC_img(scene_xmls['tgt_mask'], integrator_mask, sensor_ids, res, spp_ref)
for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
    save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/refmask_{sensor_ids[i]}.exr', res)
ref_v = transform_ek(scene.param_map[key].vertex_positions, scene.param_map[key].to_world).numpy().astype(np.double)
ref_f = scene.param_map[key].face_indices.numpy().astype(np.int64)

if start_itr != 0:
    sdf, start_itr = load_tensor(f'{outdir}/optimized/sdf.*.pt', start_itr)
    deform, start_itr = load_tensor(f'{outdir}/optimized/deform.*.pt', start_itr)
else:
    sdf = deform = None
dmtet = DMTetGeometry(64, 4, sdf, deform)
obj.write_obj(SDF_PATH, dmtet.getMesh())
scene, init_imgs = renderC_img(scene_xmls['sdf'], integrator, sensor_ids, res, 32)
scene, init_masks = renderC_img(scene_xmls['sdf_mask'], integrator_mask, sensor_ids, res, 32)
for i, (img, mask) in enumerate(zip(init_imgs, init_masks)):
    save_img(img, f'{outdir}/ref/init_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/initmask_{sensor_ids[i]}.exr', res)
del init_imgs, init_masks
to_world = scene.param_map[key].to_world.numpy().astype(np.double).squeeze()

# ek.cuda_malloc_trim()
ref_imgs = torch.stack([torch.from_numpy(img).cuda() for img in ref_imgs])
ref_masks = torch.stack([torch.from_numpy(mask).cuda() for mask in ref_masks])
trainset = DatasetMesh(sensor_ids, ref_imgs, ref_masks)
opt = Adam(dmtet.parameters(), lr)
config.key = key
config.integrator = integrator
config.integrator_mask = integrator_mask

def prepare_scene():
    scene = psdr_cuda.Scene()
    scene.load_string(scene_xmls['sdf'], False)
    scene.opts.spp = spp_opt
    scene.opts.width = res[0]
    scene.opts.height = res[1]
    scene.opts.log_level = 0
    config.scene = scene
    scene = psdr_cuda.Scene()
    scene.load_string(scene_xmls['sdf_mask'], False)
    scene.opts.spp = 1
    scene.opts.width = res[0]
    scene.opts.height = res[1]
    scene.opts.log_level = 0
    config.scene_mask = scene

prepare_scene()  
img_losses = []
mask_losses = []
reg_losses = []
distances = []
for it in tqdm(range(start_itr, start_itr + n_itr)):
    img_loss_ = mask_loss_ = reg_loss_ = 0.
    for ids, ref_imgs, ref_masks in DataLoader(trainset, batch_size, True):
        config.sensor_ids = ids

        imgs = dmtet(renderDVA)

        img_loss = loss_fn(imgs[0], ref_imgs)
        mask_loss = functional.mse_loss(imgs[1], ref_masks)
        reg_loss = sdf_reg_loss(dmtet.sdf, dmtet.all_edges).mean() \
            * (sdf_weight - (sdf_weight - 0.01) * min(1.0, it / 1000))
        loss = img_loss + mask_loss + reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        m = dmtet.getMesh()
        print('reloading mesh')
        with torch.no_grad():
            v = Vector3fD(m.v_pos)
            f = Vector3iD(m.t_pos_idx.to(torch.int32))
            config.scene.reload_mesh_mem(config.scene.param_map[key], v, f)
            config.scene_mask.reload_mesh_mem(config.scene_mask.param_map[key], v, f)
            config.scene.configure()
            config.scene_mask.configure()
            img_loss_ += img_loss
            mask_loss_ += mask_loss
            reg_loss_ += reg_loss
            if it % 2 == 0:
                for i, id in enumerate(ids):
                    save_img(imgs[0][i], f'{outdir}/train/{id}/train.{it+1:04}.png', res)
                    save_img(imgs[1][i], f'{outdir}/train/{id}/train_mask.{it+1:04}.exr', res)

    with torch.no_grad():
        if (it+1)%save_interval == 0:
            torch.save(dmtet.sdf.detach(), f'{outdir}/optimized/sdf.{it+1}.pt')
            torch.save(dmtet.deform.detach(), f'{outdir}/optimized/deform.{it+1}.pt')
        img_losses.append(img_loss_.cpu().numpy())
        mask_losses.append(mask_loss_.cpu().numpy())
        reg_losses.append(reg_loss_.cpu().numpy())
        distances.append(hausdorff(transform_np(m.v_pos.cpu().numpy(), to_world),
                m.t_pos_idx.cpu().numpy(), ref_v, ref_f))

plt.plot(img_losses, label='Image Loss')
plt.plot(mask_losses, label='Mask Loss')
plt.plot(reg_losses, label='SDF Regularization Loss')
plt.yscale('log')
plt.legend()
plt.savefig(f'{outdir}/stats/losses_itr{it+1}_lr{lr}_weight{sdf_weight}.png')
plt.close()
plt.plot(distances)
plt.ylabel("Hausdorff Distance")
plt.savefig(f'{outdir}/stats/distances_itr{it+1}_lr{lr}_weight{sdf_weight}.png')

obj.write_obj(SDF_PATH, dmtet.getMesh())
del dmtet, trainset, opt, ref_imgs, ref_masks
torch.cuda.empty_cache()
scene, opt_imgs = renderC_img(scene_xmls['sdf'], integrator, sensor_ids, res, 32)
scene_mask, opt_masks = renderC_img(scene_xmls['sdf_mask'], integrator_mask, sensor_ids, res, 32)
for i, (img, mask) in enumerate(zip(opt_imgs, opt_masks)):
    save_img(img, f'{outdir}/optimized/itr{it+1}_{i}.png', res)
    save_img(mask, f'{outdir}/optimized/itr{it+1}_{i}_mask.exr', res)
scene.param_map[key].dump(f'{outdir}/optimized/optimized_itr{it+1}_lr{lr}_weight{sdf_weight}.obj')

for id in sensor_ids:
    imgs2video(f"{outdir}/train/{id}/train.*.png", f"{outdir}/train/{id}.mp4", 10)
    imgs2video(f"{outdir}/train/{id}/train_mask.*.exr", f"{outdir}/train/{id}_mask.mp4", 10)

'''
Notes:
    - After the first 100 iterations, the surface does not change much
    during the following optimization (trapped in a local minimum)
        - Increase the res of the tet grid (too costly)
        - Increase spp
            - Significantly improved the reconstructed geometry
        - Increase res of images
'''

'''
TODO
    - Optimize code to allow for higher image resolutions
        - highest memory consumption: 4437
            - when rendering reference images
    - Remove the need to have two scenes
    - Make it possible to reload mesh from memory on the fly
    - Abstract psdr-cuda and Enoki away
'''