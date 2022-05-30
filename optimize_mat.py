import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from util import *

FLAGS = json.load(open('configs/spot_env.json', 'r'))
n_sensors, sensor_ids = FLAGS['n_sensors'], FLAGS['sensor_ids']
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
key, mat_key = FLAGS['key'], FLAGS['mat_key']
res = FLAGS['res']
loss_fn = get_loss_fn(FLAGS['loss_fn'])
lr = FLAGS['learning_rate']
if FLAGS['optimizer'] == 'adam':
    outdir = f"output/{FLAGS['name']}/adam/sensors_{FLAGS['n_sensors']}"
else:
    raise NotImplementedError
for i in range(n_sensors):
    os.makedirs(f'{outdir}/{sensor_ids[i]}', exist_ok=True)
if start_itr != 0:
    mat, start_itr = load_ckp(f'{outdir}/mat.*.pt', start_itr)
    if mat is not None:
        assert(str(mat.device) == 'cuda:0')
        assert(mat.requires_grad == False)
else:
    mat = None

integrator = psdr_cuda.DirectIntegrator(bsdf_samples=2, light_samples=2)
scene, ref_imgs = renderC_img(FLAGS['scene'], integrator, sensor_ids, res)
for i, ref in enumerate(ref_imgs):
    save_img(ref, f'{outdir}/ref_{sensor_ids[i]}.png', res)
    ref_imgs[i] = ref.torch()
ref_imgs = torch.stack(ref_imgs)
ref_tex = scene.param_map[key].bsdf.reflectance.data.torch()

scene.param_map[key].bsdf.reflectance.data = scene.param_map[mat_key].reflectance.data
scene.configure()
init_imgs = [integrator.renderC(scene, id) for id in sensor_ids]
for i, init in enumerate(init_imgs):
    save_img(init, f'{outdir}/init_{sensor_ids[i]}.png', res)
del init_imgs

if mat is None: mat = scene.param_map[mat_key].reflectance.data.torch()
mat_res = torch.tensor(scene.param_map[mat_key].reflectance.resolution)
mat.requires_grad_()
opt = Adam([mat], lr)
scene.opts.spp = 1
scene.configure()

img_losses = []
errors = []
for it in tqdm(range(start_itr, start_itr + n_itr)):
    imgs = renderDM({
            'scene': scene,
            'key': key,
            'integrator': integrator,
            'sensor_ids': sensor_ids
        }, mat)
    for i, id in enumerate(sensor_ids):
        save_img(imgs[i], f'{outdir}/{id}/train.{it+1:03}.png', res)

    img_loss = loss_fn(imgs, ref_imgs)
    loss = img_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        img_losses.append(img_loss.detach().cpu().numpy())
        errors.append(functional.mse_loss(mat, ref_tex).cpu().numpy())
        if (it+1) % save_interval == 0:
            torch.save(mat.detach(), f'{outdir}/mat.{it+1}.pt')
            save_img(mat, f'{outdir}/mat.{it+1}.png', mat_res)

plt.plot(img_losses)
plt.yscale('log')
plt.title('Image Loss')
plt.savefig(f'{outdir}/losses_itr{it+1}_lr{lr}.png')
plt.close()
plt.plot(errors)
plt.title('Parameter MSE')
plt.savefig(f'{outdir}/param_errors_itr{it+1}_lr{lr}.png')

with torch.no_grad():
    scene.opts.spp = 32
    scene.param_map[key].bsdf.reflectance.data = Vector3fD(mat)
    scene.configure()
    for i in sensor_ids:
        save_img(integrator.renderC(scene, sensor_id=i), f'{outdir}/itr{it+1}_{i}.png', res)