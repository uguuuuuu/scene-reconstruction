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
from render.mlptexture import MLPTexture3D

FLAGS = json.load(open('configs/spot_env_dmtet.json', 'r'))
sdf_weight, lr = FLAGS['sdf_weight'], FLAGS['learning_rate']
batch_size = FLAGS['batch_size']
key = FLAGS['key']
res = FLAGS['res']
spp_ref, spp_opt = FLAGS['spp_ref'], FLAGS['spp_opt']
loss_fn = get_loss_fn(FLAGS['loss_fn'], FLAGS['tonemap'])
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
outdir = f"output/{FLAGS['name']}"
SDF_PATH = f'{outdir}/optimized/sdf.obj'
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

if start_itr != 0:
    print('Loading checkpoint...')
    ckp, start_itr = load_ckp(f'{outdir}/optimized/ckp.*.tar', start_itr)
    if ckp is not None:
        print(f'Loaded checkpoint of epoch {start_itr}')
    print('Finished')
else:
    ckp = None
if ckp is None:
    ckp = {}
print('Initializing parameters...')
dmtet = DMTetGeometry(64, 4, ckp.get('sdf'), ckp.get('deform'))
kd_min, kd_max = torch.tensor([0., 0., 0.], device='cuda'), torch.tensor([1., 1., 1.], device='cuda')
material = MLPTexture3D(dmtet.getAABB(), min_max=torch.stack([kd_min, kd_max]))
if ckp.get('mat') is not None:
    material.load_state_dict(ckp['mat'])
    material.train()
dmtet.getMesh().material = material.sample(dmtet.getMesh().v_pos)
obj.write_obj(SDF_PATH, mesh.auto_normals(dmtet.getMesh()))
print(f'{dmtet.getMesh().v_pos.shape[0]} vertices')
print('Finished')
print('Rendering initial images...')
scene = Scene(scene_info['src'])
scene.reload_mesh(key, dmtet.getMesh().v_pos, dmtet.getMesh().t_pos_idx.int())
scene.reload_mat(key, dmtet.getMesh().material)
scene.set_opts(res, 32, sppe=0, sppse=0)
init_imgs = scene.renderC(integrator, sensor_ids)
init_masks = scene.renderC(integrator_mask, sensor_ids)
for i, (img, mask) in enumerate(zip(init_imgs, init_masks)):
    save_img(img, f'{outdir}/ref/init_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/initmask_{sensor_ids[i]}.exr', res)
del init_imgs, init_masks, scene
print('Finished')

ref_imgs = torch.stack([torch.from_numpy(img).cuda() for img in ref_imgs])
ref_masks = torch.stack([torch.from_numpy(mask).cuda() for mask in ref_masks])
trainset = DatasetMesh(sensor_ids, ref_imgs, ref_masks)
opt = Adam(list(dmtet.parameters())+list(material.parameters()), lr)
if ckp.get('optimizer') is not None:
    opt.load_state_dict(ckp['optimizer'])
# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda itr: lr_schedule(itr, 0))
# if ckp.get('scheduler') is not None:
#     scheduler.load_state_dict(ckp['scheduler'])

scene = Scene(scene_info['src'])
scene.reload_mesh(key, dmtet.getMesh().v_pos, dmtet.getMesh().t_pos_idx.to(torch.int32))
scene.reload_mat(key, dmtet.getMesh().material)
scene.set_opts(res, spp=spp_opt, sppe=spp_opt, sppse=spp_opt) 
img_losses = []
mask_losses = []
reg_losses = []
distances = []
for it in tqdm(range(start_itr, start_itr + n_itr)):
    img_loss_ = mask_loss_ = reg_loss_ = 0.
    for ids, ref_imgs, ref_masks in DataLoader(trainset, batch_size, True):

        imgs = dmtet(lambda v, mat: scene.renderDVAM(v, mat, key, integrator, integrator_mask, ids))

        img_loss = loss_fn(imgs[0], ref_imgs)
        mask_loss = functional.mse_loss(imgs[1], ref_masks)
        reg_loss = sdf_reg_loss(dmtet.sdf, dmtet.all_edges).mean()
        loss = img_loss + \
            mask_loss + \
            reg_loss * (sdf_weight - (sdf_weight - 0.01) * min(1.0, it / 1000))

        opt.zero_grad()
        loss.backward()
        opt.step()

        m = dmtet.getMesh()
        v = m.v_pos
        f = m.t_pos_idx
        m.material = material.sample(v)
        with torch.no_grad():
            scene.reload_mesh(key, v, f.to(torch.int32))
            scene.reload_mat(key, m.material, res_only=True)
            img_loss_ += img_loss
            mask_loss_ += mask_loss
            reg_loss_ += reg_loss
            if it % 2 == 0:
                for i, id in enumerate(ids):
                    save_img(imgs[0][i], f'{outdir}/train/{id}/train.{it+1:04}.png', res)
                    save_img(imgs[1][i], f'{outdir}/train/{id}/train_mask.{it+1:04}.exr', res)
    # scheduler.step()
    with torch.no_grad():
        if (it+1)%save_interval == 0:
            torch.save({
                'sdf': dmtet.sdf.detach(),
                'deform': dmtet.deform.detach(),
                'mat': material.state_dict(),                
                'optimizer': opt.state_dict(),
                # 'scheduler': scheduler.state_dict()
            }, f'{outdir}/optimized/ckp.{it+1}.tar')
        img_losses.append(img_loss_.cpu().numpy())
        mask_losses.append(mask_loss_.cpu().numpy())
        reg_losses.append(reg_loss_.cpu().numpy())
        v, f = scene.get_mesh(key, True)
        distances.append(hausdorff(v, f, ref_v, ref_f))

scene.reload_mesh(key, dmtet.getMesh().v_pos, dmtet.getMesh().t_pos_idx.int())
dmtet.getMesh().material = material.sample(dmtet.getMesh().v_pos)
scene.reload_mat(key, dmtet.getMesh().material)
scene.set_opts(res, 32, sppe=0, sppse=0)
opt_imgs = scene.renderC(integrator)
opt_masks = scene.renderC(integrator_mask)
for i, (img, mask) in enumerate(zip(opt_imgs, opt_masks)):
    save_img(img, f'{outdir}/optimized/itr{it+1}_{i}.png', res)
    save_img(mask, f'{outdir}/optimized/itr{it+1}_{i}_mask.exr', res)

material.cleanup()
del dmtet, material, opt, trainset, ref_imgs, ref_masks, scene
torch.cuda.empty_cache()
ek.cuda_malloc_trim()

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

for id in sensor_ids:
    imgs2video(f"{outdir}/train/{id}/train.*.png", f"{outdir}/train/{id}.mp4", 30)
    imgs2video(f"{outdir}/train/{id}/train_mask.*.exr", f"{outdir}/train/{id}_mask.mp4", 30)


ckp, _ = load_ckp(f'{outdir}/optimized/ckp.*.tar')
dmtet = DMTetGeometry(64, 4, ckp['sdf'], ckp['deform'])
kd_min, kd_max = [0., 0., 0.], [1., 1., 1.]
material = MLPTexture3D(dmtet.getAABB(),
            min_max=torch.stack([torch.tensor(kd_min, device='cuda'), torch.tensor(kd_max, device='cuda')]))
print('Loading material')
material.load_state_dict(ckp['mat'])
material.eval()
print('Loaded')

print('Extracting texture')
m = extract_texture(dmtet.getMesh(), material)
print('Finished')
print('Writing mesh')
obj.write_obj(f'{outdir}/optimized/optimized.obj', auto_normals(m), True)
print('Finished')

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
    - Internal geometry
        - Set the learning rate to be high at the beginning and then fall off exponentially 
'''

'''
TODO
    - Experiment with different spp's of the differentiablly rendered images
        - Limit is res (320, 180) spp = sppe = sppse = 4
    - Experiment with different coordinate networks for materials
    - Experiment with different shading models
'''