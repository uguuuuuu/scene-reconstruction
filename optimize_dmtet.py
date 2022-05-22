import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from igl import hausdorff
import psdr_cuda
import torch
from torch.optim import Adam
from torch.nn import functional
from torch.utils.data import DataLoader
from util import *
from geometry.dmtet import DMTetGeometry, sdf_reg_loss
from geometry import mesh
from geometry import obj
from dataset import DatasetMesh
from render.loss import get_loss_fn
from render.scene import Scene

FLAGS = json.load(open('configs/bunny_env_largesteps_dmtet.json', 'r'))
batch_size = FLAGS['batch_size']
key = FLAGS['key']
res = FLAGS['res']
spp_ref, spp_opt = FLAGS['spp_ref'], FLAGS['spp_opt']
loss_fn = get_loss_fn(FLAGS['loss_fn'], FLAGS['tonemap'])
sdf_weight = FLAGS['sdf_weight']
lr = FLAGS['learning_rate']
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
outdir = f"output/{FLAGS['name']}"
SDF_PATH = f'{outdir}/optimized/sdf.obj'
scene_info = preprocess_scene(f"{FLAGS['scene']}.xml", sdf_path=SDF_PATH)
n_sensors = scene_info['n_sensors']
sensor_ids = range(n_sensors)

for i in range(n_sensors):
    os.makedirs(f'{outdir}/train/{sensor_ids[i]}', exist_ok=True)
os.makedirs(f'{outdir}/stats', exist_ok=True)
os.makedirs(f'{outdir}/ref', exist_ok=True)
os.makedirs(f'{outdir}/optimized', exist_ok=True)

integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')
scene, ref_imgs = renderC_img(scene_info['tgt'], integrator, res=res, spp=spp_ref)
scene, ref_masks = renderC_img(scene_info['tgt'], integrator_mask, res=res, spp=spp_ref)
for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
    save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/refmask_{sensor_ids[i]}.exr', res)
ref_v, ref_f = scene.get_mesh(key, True)

if start_itr != 0:
    sdf, start_itr = load_tensor(f'{outdir}/optimized/sdf.*.pt', start_itr)
    deform, start_itr = load_tensor(f'{outdir}/optimized/deform.*.pt', start_itr)
else:
    sdf = deform = None
dmtet = DMTetGeometry(64, 4, sdf, deform)
obj.write_obj(SDF_PATH, mesh.auto_normals(dmtet.getMesh()))
scene, init_imgs = renderC_img(scene_info['sdf'], integrator, sensor_ids, res, 32)
scene, init_masks = renderC_img(scene_info['sdf'], integrator_mask, sensor_ids, res, 32)
for i, (img, mask) in enumerate(zip(init_imgs, init_masks)):
    save_img(img, f'{outdir}/ref/init_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/initmask_{sensor_ids[i]}.exr', res)
del init_imgs, init_masks

# ek.cuda_malloc_trim()
ref_imgs = torch.stack([torch.from_numpy(img).cuda() for img in ref_imgs])
ref_masks = torch.stack([torch.from_numpy(mask).cuda() for mask in ref_masks])
trainset = DatasetMesh(sensor_ids, ref_imgs, ref_masks)
opt = Adam(dmtet.parameters(), lr)

scene = Scene(scene_info['sdf'])
scene.set_opts(res, spp=spp_opt, sppe=spp_opt, sppse=spp_opt) 
img_losses = []
mask_losses = []
reg_losses = []
distances = []
for it in tqdm(range(start_itr, start_itr + n_itr)):
    img_loss_ = mask_loss_ = reg_loss_ = 0.
    for ids, ref_imgs, ref_masks in DataLoader(trainset, batch_size, True):

        imgs = dmtet(lambda v: scene.renderDVA(v, key, integrator, integrator_mask, ids))

        img_loss = loss_fn(imgs[0], ref_imgs)
        mask_loss = functional.mse_loss(imgs[1], ref_masks)
        reg_loss = sdf_reg_loss(dmtet.sdf, dmtet.all_edges).mean() \
            * (sdf_weight - (sdf_weight - 0.01) * min(1.0, it / 5000))
        loss = img_loss + mask_loss + reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        m = dmtet.getMesh()
        with torch.no_grad():
            v = m.v_pos
            f = m.t_pos_idx.to(torch.int32)
            scene.reload_mesh(key, v, f)
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
        v, f = scene.get_mesh(key, True)
        distances.append(hausdorff(v, f, ref_v, ref_f))

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

del dmtet, trainset, opt, ref_imgs, ref_masks
torch.cuda.empty_cache()

scene.set_opts(res, 32, sppe=0, sppse=0)
opt_imgs = scene.renderC(integrator)
opt_masks = scene.renderC(integrator_mask)
for i, (img, mask) in enumerate(zip(opt_imgs, opt_masks)):
    save_img(img, f'{outdir}/optimized/itr{it+1}_{i}.png', res)
    save_img(mask, f'{outdir}/optimized/itr{it+1}_{i}_mask.exr', res)
scene.dump(key, f'{outdir}/optimized/optimized_itr{it+1}_lr{lr}_weight{sdf_weight}.obj')

for id in sensor_ids:
    imgs2video(f"{outdir}/train/{id}/train.*.png", f"{outdir}/train/{id}.mp4", 30)
    imgs2video(f"{outdir}/train/{id}/train_mask.*.exr", f"{outdir}/train/{id}_mask.mp4", 30)

'''
Issues:
    - After the first 100 iterations, the surface does not change much
    during the following optimization (trapped in a local minimum)
        - Increase res of tet grid (too costly)
            - Currently 64
        - Increase spp and res of reference images
            - Significantly improved the reconstructed geometry
            - Highest possible is res (320, 180) and spp 128
        - Train for more iterations (nvdiffrec trains for 5,000 iterations)
    - The geomtry is entangled
        - Increase spp of differentiablly rendered images
    - Too many texture coordinates (exceeding the capacity of int to index)
        - modify the uv mapping algorithm
'''

'''
TODO
    - Experiment with different spp's of the differentiablly rendered images
        - Limit is res (320, 180) spp 8
'''