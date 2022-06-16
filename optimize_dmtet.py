import os
import random
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
from render.util import scale_img
from xml_util import keep_sensors, preprocess_scene

FLAGS = json.load(open('configs/nerf_synthetic/chair_dmtet.json', 'r'))
sdf_weight, lr = FLAGS['sdf_weight'], FLAGS['learning_rate']
tet_res, tet_scale = FLAGS['tet_res'], FLAGS['tet_scale']
batch_size = FLAGS['batch_size']
key = FLAGS['key']
res, env_res = FLAGS['res'], FLAGS['env_res']
spp_ref, spp_opt = FLAGS.get('spp_ref'), FLAGS['spp_opt']
loss_fn = get_loss_fn(FLAGS['loss_fn'], FLAGS['tonemap'])
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
shading_model = FLAGS['shading_model']
ref_dir = FLAGS.get('ref_dir')
outdir = f"output/{FLAGS['name']}"
SDF_PATH = f'{outdir}/optimized/sdf.obj'
scene_info = preprocess_scene(f"{FLAGS['scene_file']}")

if start_itr != 0:
    print('Loading checkpoint...')
    ckp, start_itr = load_ckp(f'{outdir}/optimized/ckp.*.tar', start_itr)
    if ckp is not None:
        sensor_ids = ckp['sensor_ids']
        n_sensors = len(sensor_ids)
        scene_info['src'] = keep_sensors(scene_info['src'], sensor_ids)
        print(f'Loaded checkpoint of epoch {start_itr}')
    print('Finished')
else:
    ckp = None
if ckp is None:
    ckp = {}

if len(ckp.keys()) == 0:
    if FLAGS.get('n_sensors') is None:
        n_sensors = scene_info['n_sensors']
        sensor_ids = range(n_sensors)
    else:
        sensor_ids = [0]
        sensor_ids.extend(random.sample(range(1, scene_info['n_sensors']), FLAGS['n_sensors']-1))
        sensor_ids.sort()
        scene_info['src'] = keep_sensors(scene_info['src'], sensor_ids)
        n_sensors = len(sensor_ids)

for i in range(n_sensors):
    os.makedirs(f'{outdir}/train/{sensor_ids[i]}', exist_ok=True)
os.makedirs(f'{outdir}/stats', exist_ok=True)
os.makedirs(f'{outdir}/ref', exist_ok=True)
os.makedirs(f'{outdir}/optimized', exist_ok=True)

integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
integrator.hide_emitters = True
integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')
ref_v = ref_f = None
print('Writing reference images...')
if ref_dir is None:
    scene, ref_imgs = renderC_img(scene_info['tgt'], integrator, res=res, spp=spp_ref)
    scene, ref_masks = renderC_img(scene_info['tgt'], integrator_mask, res=res, spp=spp_ref)
    for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
        save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.png', res)
        save_img(mask, f'{outdir}/ref/refmask_{sensor_ids[i]}.exr', res)
    ref_masks = np.mean(ref_masks, axis=-1, keepdims=True)
    ref_imgs = np.concatenate([ref_imgs, ref_masks], axis=-1)
    del ref_masks
    ref_v, ref_f = scene.get_mesh(key, True)
else:
    filenames = next(os.walk(ref_dir), (None, None, []))[2]
    if len(filenames) == 0:
        raise FileNotFoundError('Nothing in the reference image directory!')
    def extract_number(key: str):
        number = ''
        for c in key:
            if c.isdigit():
                number += c
        return int(number)
    filenames.sort(key=extract_number)

    ref_imgs = []
    for id in sensor_ids:
        filename = filenames[id]
        filename = os.path.join(ref_dir, filename)
        img = load_img(filename)
        if tuple(img.shape[:-1]) != (res[1], res[0]):
            img = scale_img(img, (res[1], res[0])).numpy()
        ref_imgs.append(img.reshape(-1,img.shape[-1]))

    for i, img in enumerate(ref_imgs):
        save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.exr', res)
    ref_imgs = np.array(ref_imgs)
print('Finished')

print('Initializing parameters...')
dmtet = DMTetGeometry(tet_res, tet_scale, ckp.get('sdf'), ckp.get('deform'))
if shading_model == 'diffuse':
    mat_min = torch.zeros(3, dtype=torch.float32, device='cuda')
    mat_max = torch.ones(3, dtype=torch.float32, device='cuda')
else:
    mat_min = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda')
    mat_max = torch.tensor([1., 1., 10., 10., 10., 10., 10., 10., 1., 1., 1.], device='cuda')
material = MLPTexture3D(dmtet.getAABB(), min_max=torch.stack([mat_min, mat_max]), channels=mat_min.size(0))
if ckp.get('mat') is not None:
    material.load_state_dict(ckp['mat'])
    material.train()
dmtet.getMesh().material = material.sample(dmtet.getMesh().v_pos)
obj.write_obj(SDF_PATH, mesh.auto_normals(dmtet.getMesh()))
if ckp.get('envmap') is not None:
    envmap = ckp['envmap']
    envmap.requires_grad_()
else:
    envmap = torch.rand([env_res[1]*env_res[0], 3], dtype=torch.float32, device='cuda', requires_grad=True)
print(f'{dmtet.getMesh().v_pos.shape[0]} vertices')
print('Finished')

print('Rendering initial images...')
scene = Scene(scene_info['src'])
scene.reload_mesh(key, dmtet.getMesh().v_pos, dmtet.getMesh().t_pos_idx.int())
scene.reload_mat(key, dmtet.getMesh().material)
scene.reload_envmap(envmap.reshape(env_res[1], env_res[0], 3))
scene.set_opts(res, 32, sppe=0, sppse=0)
init_imgs = scene.renderC(integrator)
init_masks = scene.renderC(integrator_mask)
for i, (img, mask) in enumerate(zip(init_imgs, init_masks)):
    save_img(img, f'{outdir}/ref/init_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/ref/initmask_{sensor_ids[i]}.exr', res)
save_img(envmap, f'{outdir}/ref/initenv.exr', env_res)
del init_imgs, init_masks, scene
print('Finished')

ref_imgs = torch.stack([torch.from_numpy(img).cuda() for img in ref_imgs])
trainset = DatasetMesh(range(n_sensors), ref_imgs)
opt = Adam(list(dmtet.parameters())+list(material.parameters())+[envmap], lr)
if ckp.get('optimizer') is not None:
    opt.load_state_dict(ckp['optimizer'])
# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda itr: lr_schedule(itr, 0))
# if ckp.get('scheduler') is not None:
#     scheduler.load_state_dict(ckp['scheduler'])

scene = Scene(scene_info['src'])
scene.reload_mesh(key, dmtet.getMesh().v_pos, dmtet.getMesh().t_pos_idx.to(torch.int32))
scene.reload_mat(key, dmtet.getMesh().material)
scene.reload_envmap(envmap.reshape(env_res[1], env_res[0], 3))
scene.set_opts(res, spp=spp_opt, sppe=spp_opt)
img_losses = []
mask_losses = []
reg_losses = []
distances = []
for it in tqdm(range(start_itr, start_itr + n_itr)):
    img_loss_ = mask_loss_ = reg_loss_ = 0.
    for ids, ref_imgs in DataLoader(trainset, batch_size, True):

        m = dmtet.getMesh()
        d_mat = material.sample(m.v_pos + torch.normal(mean=0, std=0.01, size=m.v_pos.shape, device="cuda")) \
            - m.material
        reg_mat = torch.mean(d_mat[...,-3:]) * 0.03 * min(1.0, it / 500)

        imgs = dmtet(lambda v, mat: scene.renderDVAME(v, mat, envmap, key, integrator, integrator_mask, ids))

        img_loss = loss_fn(imgs[...,:3], ref_imgs[...,:3])
        mask_loss = functional.mse_loss(imgs[...,3], ref_imgs[...,3])
        reg_sdf = sdf_reg_loss(dmtet.sdf, dmtet.all_edges).mean()
        white = envmap.mean(dim=-1, keepdim=True)
        reg_lgt = functional.l1_loss(envmap, white.expand(-1,3))
        loss = img_loss + \
            mask_loss + \
            reg_sdf * (sdf_weight - (sdf_weight - 0.01) * min(1.0, it / 1000)) + \
            reg_lgt + \
            reg_mat

        opt.zero_grad()
        loss.backward()
        opt.step()

        m = dmtet.getMesh()
        v = m.v_pos
        f = m.t_pos_idx
        m.material = material.sample(v)
        with torch.no_grad():
            envmap.clamp_min_(0.)
            scene.reload_mesh(key, v, f.to(torch.int32))
            scene.reload_mat(key, m.material, res_only=True)
            img_loss_ += img_loss
            mask_loss_ += mask_loss
            reg_loss_ += reg_sdf
            if it % 2 == 0:
                for i, id in enumerate(ids):
                    save_img(imgs[i,:,:3], f'{outdir}/train/{sensor_ids[id]}/train.{it+1:04}.exr', res)
                    save_img(imgs[i,:,3:], f'{outdir}/train/{sensor_ids[id]}/train_mask.{it+1:04}.exr', res)
    # scheduler.step()
    with torch.no_grad():
        if (it+1)%save_interval == 0:
            torch.save({
                'sdf': dmtet.sdf.detach(),
                'deform': dmtet.deform.detach(),
                'mat': material.state_dict(), 
                'envmap': envmap.detach(),
                'env_res': env_res,               
                'optimizer': opt.state_dict(),
                'sensor_ids': sensor_ids,
                # 'scheduler': scheduler.state_dict()
            }, f'{outdir}/optimized/ckp.{it+1}.tar')
        img_losses.append(img_loss_.cpu().numpy())
        mask_losses.append(mask_loss_.cpu().numpy())
        reg_losses.append(reg_loss_.cpu().numpy())
        if ref_v is not None and ref_f is not None:
            v, f = scene.get_mesh(key, True)
            distances.append(hausdorff(v, f, ref_v, ref_f))

scene.reload_mesh(key, dmtet.getMesh().v_pos, dmtet.getMesh().t_pos_idx.int())
dmtet.getMesh().material = material.sample(dmtet.getMesh().v_pos)
scene.reload_mat(key, dmtet.getMesh().material)
scene.reload_envmap(envmap.reshape(env_res[1], env_res[0], 3))
scene.set_opts(res, 32, sppe=0, sppse=0)
opt_imgs = scene.renderC(integrator)
opt_masks = scene.renderC(integrator_mask)
for i, (img, mask) in enumerate(zip(opt_imgs, opt_masks)):
    save_img(img, f'{outdir}/optimized/itr{it+1}_{sensor_ids[i]}.png', res)
    save_img(mask, f'{outdir}/optimized/itr{it+1}_{sensor_ids[i]}_mask.exr', res)

material.cleanup()
del dmtet, material, opt, trainset, ref_imgs, scene
torch.cuda.empty_cache()
ek.cuda_malloc_trim()

plt.plot(img_losses, label='Image Loss')
plt.plot(mask_losses, label='Mask Loss')
plt.plot(reg_losses, label='SDF Regularization Loss')
plt.yscale('log')
plt.legend()
plt.savefig(f'{outdir}/stats/losses_itr{it+1}_lr{lr}_weight{sdf_weight}.png')
plt.close()
if len(distances) != 0:
    plt.plot(distances)
    plt.ylabel("Hausdorff Distance")
    plt.savefig(f'{outdir}/stats/distances_itr{it+1}_lr{lr}_weight{sdf_weight}.png')

for id in sensor_ids:
    imgs2video(f"{outdir}/train/{id}/train.*.exr", f"{outdir}/train/{id}.mp4", 30)
    imgs2video(f"{outdir}/train/{id}/train_mask.*.exr", f"{outdir}/train/{id}_mask.mp4", 30)


'''
Issues:
    - Using the current shading model (rough conductor) causes the material gradients to be too large
    sometimes (-/+inf), causing the program to crash and producing low-quality gradients
        - current solution: log-tonemap before calculating loss + smoothness regularizer + clamp
        - things to try:
            - increase the batch size
            - increase spp
            - log-tonemap before calculating loss
            - use smoothness regularizer to regularize the material
            - decrease learning rate
'''

'''
TODO
    - Experiment with different spp's of the differentiablly rendered images
        - Limit is res (320, 180) spp = sppe = sppse = 4
    - Experiment with different coordinate networks for materials
    - Experiment with different shading models
    - Experiment with different datasets
    - Validate after training
    - Implement a denoiser
    - demodulate lighting: separate lighting and reflectance
'''