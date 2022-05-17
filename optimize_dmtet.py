import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from tqdm import tqdm
import matplotlib.pyplot as plt
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector3i as Vector3iD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
import torch
from torch.optim import Adam
from torch.nn import functional
from utils import *
import config
from geometry.dmtet import DMTetGeometry, sdf_reg_loss
from geometry import obj

res = (284, 216)
n_sensors = 2
sensor_ids = [2, 13]
outdir = f"output/dmtet/spot_env"
key = 'Mesh[0]'
target_scene = 'data/scenes/spot_env/spot_env.xml'
source_scene = 'data/scenes/spot_env/spot_env_dmtet.xml'
sdf_weight = 0.2
lr = 0.01
n_itr, start_itr, save_interval = 500, -1, 500
for i in range(n_sensors):
    os.makedirs(f'{outdir}/{sensor_ids[i]}', exist_ok=True)
sdf, start_itr = load_tensor(f'{outdir}/sdf.*.pt', start_itr)
deform, start_itr = load_tensor(f'{outdir}/deform.*.pt', start_itr)
assert(not sdf.requires_grad if sdf is not None else True)
assert(not deform.requires_grad if deform is not None else True)
dmtet = DMTetGeometry(32, 1, sdf, deform)

integrator = psdr_cuda.DirectIntegrator(bsdf_samples=1, light_samples=1)
scene, ref_imgs = renderC_img(target_scene, integrator, sensor_ids, res)
for i, ref in enumerate(ref_imgs):
    save_img(ref, f'{outdir}/ref_{sensor_ids[i]}.png', res)
    ref_imgs[i] = ref.torch()
ref_imgs = torch.stack(ref_imgs)
ref_v = scene.param_map[key].vertex_positions.numpy()
ref_f = scene.param_map[key].face_indices.numpy()

m = dmtet.getMesh()
obj.write_obj(f'{outdir}/mesh.obj', m)
scene = psdr_cuda.Scene()
scene.load_file(source_scene, False)
scene.opts.spp = 32
scene.opts.width = res[0]
scene.opts.height = res[1]
scene.configure()
init_imgs = [integrator.renderC(scene, id) for id in sensor_ids]
for i, init in enumerate(init_imgs):
    save_img(init, f'{outdir}/init_{sensor_ids[i]}.png', res)
del init_imgs

opt = Adam(dmtet.parameters(), lr)
config.key = key
config.integrator = integrator
config.sensor_ids = sensor_ids

img_losses = []
reg_losses = []
for it in tqdm(range(start_itr, start_itr + n_itr)):
    m = dmtet.getMesh()
    obj.write_obj(f'{outdir}/mesh.obj', m)
    scene = psdr_cuda.Scene()
    scene.load_file(source_scene, False)
    scene.opts.spp = 1
    scene.opts.width = res[0]
    scene.opts.height = res[1]
    scene.opts.log_level = 0
    config.scene = scene

    imgs = dmtet(renderDV)
    img_loss = functional.l1_loss(imgs, ref_imgs)
    reg_loss: torch.Tensor = sdf_reg_loss(dmtet.sdf, dmtet.all_edges).mean() \
        * (sdf_weight - (sdf_weight - 0.01) * min(1.0, it / 5000))
    loss = img_loss + reg_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        for i, img in enumerate(imgs):
            save_img(img, f'{outdir}/{sensor_ids[i]}/train.{it+1:03}.png', res)
        img_losses.append(img_loss.detach().cpu().numpy())
        reg_losses.append(reg_loss.detach().cpu().numpy())
        if (it+1)%save_interval == 0:
            torch.save(dmtet.sdf.detach(), f'{outdir}/sdf.{it+1}.pt')
            torch.save(dmtet.deform.detach(), f'{outdir}/deform.{it+1}.pt')

plt.plot(img_losses, label='Image Loss')
plt.plot(reg_losses, label='SDF Regularization Loss')
plt.legend()
plt.savefig(f'{outdir}/losses.png')

m = dmtet.getMesh()
obj.write_obj(f'{outdir}/mesh.obj', m)
scene = psdr_cuda.Scene()
scene.load_file(source_scene, False)
scene.opts.spp = 32
scene.opts.width = 284
scene.opts.height = 216
scene.configure()
final_imgs = [integrator.renderC(scene, id) for id in sensor_ids]
for i, img in enumerate(final_imgs):
    save_img(img, f'{outdir}/itr{it+1}_{sensor_ids[i]}.png', res)