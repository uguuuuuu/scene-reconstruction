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
from util import *
from render.loss import get_loss_fn
from render.util import scale_img
from xml_util import keep_sensors, preprocess_scene
from dataset import DatasetMesh
from geometry.util import laplacian_uniform, regularization_loss, remesh
from denoise import oidn
from denoise import svgf

EPSILON = 1e-3

FLAGS = json.load(open('configs/nerf_synthetic/drums.json', 'r'))
batch_size = FLAGS['batch_size']
start_itr, n_itr, save_interval = FLAGS['start_itr'], FLAGS['n_itr'], FLAGS['save_interval']
remesh_interval = FLAGS.get('remesh_interval')
if remesh_interval < 0: remesh_interval = None
key = FLAGS['key']
res = FLAGS['res']
spp_ref, spp_opt = FLAGS.get('spp_ref'), FLAGS['spp_opt']
loss_fn = get_loss_fn(FLAGS['loss_fn'], FLAGS['tonemap'])
lambda_, alpha_ = FLAGS['lambda'], FLAGS['alpha']
if alpha_ == 0: alpha_ = None
lr = FLAGS['learning_rate']
shading_model = FLAGS['shading_model']
ref_dir = FLAGS.get('ref_dir')
outdir = f"output/{FLAGS['name']}"
scene_info = preprocess_scene(f"{FLAGS['scene_file']}", shading_model=shading_model)

try:
    sensor_ids = np.load('data/sensor_ids.npy')
    n_sensors = len(sensor_ids)
    scene_info['src'] = keep_sensors(scene_info['src'], sensor_ids)
except FileNotFoundError:
    n_sensors = scene_info['n_sensors']
    sensor_ids = range(n_sensors)

for i in range(n_sensors):
    os.makedirs(f'{outdir}/train/{sensor_ids[i]}', exist_ok=True)
os.makedirs(f'{outdir}/stats', exist_ok=True)
os.makedirs(f'{outdir}/ref', exist_ok=True)
os.makedirs(f'{outdir}/optimized', exist_ok=True)

ref_v = ref_f = None
print('Writing reference images...')
if ref_dir is None:
    # Rendering training images
    scene, ref_imgs = renderC_img(scene_info['tgt'], res=res, spp=spp_ref)
    scene, ref_masks = renderC_img(scene_info['tgt'], res=res, spp=spp_ref)
    for i, (img, mask) in enumerate(zip(ref_imgs, ref_masks)):
        save_img(img, f'{outdir}/ref/ref_{sensor_ids[i]}.png', res)
        save_img(mask, f'{outdir}/ref/refmask_{sensor_ids[i]}.exr', res)
    ref_masks = np.mean(ref_masks, axis=-1, keepdims=True)
    ref_imgs = np.concatenate([ref_imgs, ref_masks], axis=-1)
    ref_v, ref_f = scene.get_mesh(key, True)
else:
    # Read existing training images
    filenames = next(os.walk(ref_dir), (None, None, []))[2]
    if len(filenames) == 0:
        raise FileNotFoundError(f'Reference images not found in {ref_dir}!')
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

print('Rendering initial images...')
scene = Scene(scene_info['src'])
if shading_model == 'diffuse':
    scene.reload_mat(key, load_img('data/meshes/texture_kd.exr'))
elif shading_model == 'specular':
    specular_reflectance = load_img('data/meshes/texture_kd.exr')
    mat_res = (specular_reflectance.shape[1], specular_reflectance.shape[0])
    specular_reflectance = torch.ones([mat_res[1], mat_res[0], 3], dtype=torch.float32, device='cuda')
    alpha = torch.ones([mat_res[1], mat_res[0], 1], dtype=torch.float32, device='cuda') * 0.5
    eta = torch.ones_like(specular_reflectance) * 1.5
    k = torch.zeros_like(specular_reflectance)
    mat = torch.cat([alpha, eta, k, specular_reflectance], dim=-1)
    scene.reload_mat(key, mat)
else:
    raise NotImplementedError('Unsupported BSDF: ' + shading_model)
scene.reload_envmap(load_img('data/envmaps/source_envmap.exr'))
scene.set_opts(res, 32, sppe=0, sppse=0)
init_imgs = scene.renderC(img_type='shaded')
init_masks = scene.renderC(img_type='mask')
for i, (img, mask) in enumerate(zip(init_imgs, init_masks)):
    save_img(img, f'{outdir}/ref/init_{sensor_ids[i]}.exr', res)
    save_img(mask, f'{outdir}/ref/initmask_{sensor_ids[i]}.exr', res)
del init_imgs, init_masks, scene
print("Finished")

print('Loading checkpoint...')
if start_itr != 0:
    ckp, start_itr = load_ckp(f'{outdir}/optimized/ckp.*.tar', start_itr)
    if ckp is not None:
        print(f'Loaded checkpoint from epoch {start_itr}')
    else:
        ckp = {}
else:
    ckp = {}
print('Finished')

scene = Scene(scene_info['src'])
print('Initializing parameters...')
if ckp.get('vertex_positions') is None: 
    v, f, uvs, uv_idx = scene.get_mesh(key, return_uv=True)
    v, f = torch.from_numpy(v).cuda(), torch.from_numpy(f).cuda()
else:
    v = ckp['vertex_positions']
    f = ckp['faces']
    uvs = ckp['uvs']
    uv_idx = ckp['uv_idx']
with torch.no_grad():
    M = compute_matrix(v, f, lambda_, alpha=alpha_)
    u: torch.Tensor = to_differential(M, v)
    L = laplacian_uniform(v.shape[0], f)
u.requires_grad_()
if shading_model == 'diffuse':
    mat_min = torch.zeros(3, dtype=torch.float32, device='cuda') + EPSILON
    mat_max = torch.ones(3, dtype=torch.float32, device='cuda')
elif shading_model == 'specular':
    mat_min = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda') + EPSILON
    mat_max = torch.tensor([1., 5., 5., 5., 5., 5., 5., 1., 1., 1.], device='cuda')
else:
    raise NotImplementedError('Unsupported BSDF: ' + shading_model)
if ckp.get('mat') is None:
    if shading_model == 'diffuse':
        texture = torch.from_numpy(load_img('data/meshes/texture_kd.exr')).cuda()
        mat_min = torch.zeros(3, dtype=torch.float32, device='cuda') + EPSILON
        mat_max = torch.ones(3, dtype=torch.float32, device='cuda')
    elif shading_model == 'specular':
        specular_reflectance = load_img('data/meshes/texture_kd.exr')
        mat_res = (specular_reflectance.shape[1], specular_reflectance.shape[0])
        specular_reflectance = torch.ones([mat_res[1], mat_res[0], 3], dtype=torch.float32, device='cuda')
        alpha = torch.ones([mat_res[1], mat_res[0], 1], dtype=torch.float32, device='cuda') * 0.5
        eta = torch.ones_like(specular_reflectance) * 1.5
        k = torch.zeros_like(specular_reflectance)
        texture = torch.cat([alpha, eta, k, specular_reflectance], dim=-1)
        mat_min = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda') + EPSILON
        mat_max = torch.tensor([1., 5., 5., 5., 5., 5., 5., 1., 1., 1.], device='cuda')
    else:
        raise NotImplementedError('Unsupported BSDF: ' + shading_model)

    mat_res = texture.shape[:2]
    mat_res = (mat_res[1], mat_res[0])
    texture = texture.reshape(mat_res[1]*mat_res[0],-1).cuda().contiguous()
else:
    texture = ckp['mat']
    mat_res = ckp['mat_res']
if ckp.get('envmap') is None:
    envmap = load_img('data/envmaps/source_envmap.exr')
    env_res = envmap.shape[:2]
    env_res = (env_res[1], env_res[0])
    envmap = torch.from_numpy(envmap).reshape(-1,3).cuda().contiguous()
else:
    envmap = ckp['envmap']
    env_res = ckp['env_res']
texture.requires_grad_()
envmap.requires_grad_()
print('Finished')

opt_mat = Adam([texture], lr)
if ckp.get('optimizer_mat') is not None:
    opt_mat.load_state_dict(ckp['optimizer_mat'])
opt_geom = AdamUniform([u], lr * 1.5)
if ckp.get('optimizer_geom') is not None:
    opt_geom.load_state_dict(ckp['optimizer_geom'])
opt_env = Adam([envmap], lr * 2)
if ckp.get('optimizer_env') is not None:
    opt_env.load_state_dict(ckp['optimizer_env'])

# denoiser = oidn.load_denoiser('hdr')

scene.reload_mesh(key, v, f, uvs, uv_idx)
scene.reload_mat(key, texture.reshape(mat_res[1], mat_res[0], -1))
scene.reload_envmap(envmap.reshape(env_res[1], env_res[0], -1))
scene.set_opts(res, spp=spp_opt, sppe=spp_opt, sppse=spp_opt)
ref_imgs = torch.stack([torch.from_numpy(img).cuda() for img in ref_imgs])
trainset = DatasetMesh(range(n_sensors), ref_imgs)
img_losses = []
mask_losses = []
lgt_losses = []
smooth_losses = []
distances = []
for it in tqdm(range(start_itr, start_itr + n_itr)):
    img_loss_ = mask_loss_ = lgt_loss_ = 0.
    for ids, ref_imgs in DataLoader(trainset, batch_size, True):
        v = from_differential(M, u)

        imgs_demod, imgs_mask, imgs_alb = scene.renderD_demod(v, texture, envmap, key, ids)

        imgs_demod = imgs_demod.reshape(batch_size, res[1], res[0], 3)

        imgs_denoised = imgs_demod
        # imgs_denoised = denoiser(imgs_demod)
        # imgs_denoised = lerp(imgs_demod, imgs_denoised, min(1, it / 1000))

        imgs_demod = imgs_demod.reshape(batch_size, res[1]*res[0], 3)
        imgs_denoised = imgs_denoised.reshape(batch_size, res[1]*res[0], 3)
        imgs = imgs_denoised * imgs_alb

        img_loss = loss_fn(imgs, ref_imgs[...,:3])
        mask_loss = functional.mse_loss(torch.mean(imgs_mask, dim=-1), ref_imgs[...,3])
        # white = envmap.mean(dim=-1, keepdim=True)
        # reg_loss_lgt = functional.l1_loss(envmap, white.expand(-1,3))
        reg_lgt = lgt_reg_loss(imgs_demod, ref_imgs[...,:3])
        loss = img_loss \
            + mask_loss \
            + reg_lgt * 0.15 \
            # + reg_mat * 0.1

        opt_mat.zero_grad()
        opt_geom.zero_grad()
        opt_env.zero_grad()
        loss.backward()
        opt_mat.step()
        opt_geom.step()
        opt_env.step()

        with torch.no_grad():
            texture.clamp_(mat_min, mat_max)
            envmap.clamp_min_(0.)
            img_loss_ += img_loss
            mask_loss_ += mask_loss
            lgt_loss_ += reg_lgt
            if it % 4 == 0:
                for i, id in enumerate(ids):
                    save_img(imgs[i], f'{outdir}/train/{sensor_ids[id]}/train.{it+1:04}.exr', res)
                    save_img(imgs_demod[i], f'{outdir}/train/{sensor_ids[id]}/train_demod.{it+1:04}.exr', res)
                    save_img(imgs_mask[i], f'{outdir}/train/{sensor_ids[id]}/train_mask.{it+1:04}.exr', res)
                    save_img(imgs_alb[i], f'{outdir}/train/{sensor_ids[id]}/train_alb.{it+1:04}.exr', res)
                    save_img(imgs_denoised[i], f'{outdir}/train/{sensor_ids[id]}/train_denoised.{it+1:04}.exr', res)
                save_img(envmap, f'{outdir}/train/train_envmap.{it+1:04}.exr', env_res)
    with torch.no_grad():
        if (it+1) % save_interval == 0:
            _, f, uvs, uv_idx = scene.get_mesh(key, return_uv=True)
            torch.save({
                'vertex_positions': v.detach(),
                'faces': torch.from_numpy(f).cuda(),
                'uvs': torch.from_numpy(uvs).cuda(),
                'uv_idx': torch.from_numpy(uv_idx).cuda(),
                'mat': texture.detach(),
                'mat_res': mat_res,
                'envmap': envmap.detach(),
                'env_res': env_res,
                'optimizer_mat': opt_mat.state_dict(),
                'optimizer_geom': opt_geom.state_dict(),
                'optimizer_env': opt_env.state_dict()
            }, f'{outdir}/optimized/ckp.{it+1}.tar')
        img_losses.append(img_loss_.cpu().numpy())
        mask_losses.append(mask_loss_.cpu().numpy())
        lgt_losses.append(lgt_loss_.cpu().numpy())
        smooth_losses.append(regularization_loss(L, v, True).cpu().numpy())
        if ref_v is not None and ref_f is not None:
            v, f = scene.get_mesh(key, True)
            distances.append(hausdorff(v, f, ref_v, ref_f))
    # Remesh
    if remesh_interval is not None and (it+1) % remesh_interval == 0 and (it+1 - start_itr) != n_itr:
        with torch.no_grad():
            v = from_differential(M, u)
            _, f = scene.get_mesh(key)
            v, f = remesh(v, torch.from_numpy(f).cuda())
            _, uv_idx, uvs = xatlas.parametrize(v.cpu().numpy(), f.cpu().numpy())
            scene.reload_mesh(key, v, f.int(), uvs, uv_idx.astype(np.int32))
            scene.set_opts(res, spp=32, sppe=0, sppse=0)
            remesh_imgs = scene.renderC('shaded')
            remesh_masks = scene.renderC('mask')
            for i, id in enumerate(sensor_ids):
                save_img(remesh_imgs[i], f'{outdir}/ref/remesh_{id}.png', res)
                save_img(remesh_masks[i], f'{outdir}/ref/remeshmask_{id}.exr', res)
            scene = Scene(scene_info['src'])
            scene.reload_mesh(key, v, f.int(), uvs, uv_idx.astype(np.int32))
            scene.reload_mat(key, texture.reshape(mat_res[1], mat_res[0], -1))
            scene.reload_envmap(envmap.reshape(env_res[1], env_res[0], -1))
            scene.set_opts(res, spp=spp_opt, sppe=spp_opt, sppse=spp_opt)
            v, f = scene.get_mesh(key)
            v, f = torch.from_numpy(v).cuda(), torch.from_numpy(f).cuda()
            M = compute_matrix(v, f, lambda_, alpha=alpha)
            u: torch.Tensor = to_differential(M, v)
            L = laplacian_uniform(v.shape[0], f)
        u.requires_grad_()
        opt_mat = Adam([texture, envmap], lr)
        opt_geom = AdamUniform([u], lr * 1.5)
        opt_env = Adam([envmap], lr * 2)

'''
Save results
'''
plt.plot(img_losses, label='Image Loss')
plt.plot(mask_losses, label='Mask Loss')
plt.yscale('log')
plt.legend()
plt.savefig(f'{outdir}/stats/losses_itr{it+1}_l{lambda_}_lr{lr}.png')
plt.close()
plt.plot(lgt_losses, label='Light Loss')
plt.plot(smooth_losses, label='Smoothness Loss')
plt.yscale('log')
plt.legend()
plt.savefig(f'{outdir}/stats/losses_reg_itr{it+1}_l{lambda_}_lr{lr}.png')
plt.close()
if len(distances) != 0:
    plt.plot(distances)
    plt.ylabel("Hausdorff Distance")
    plt.savefig(f'{outdir}/stats/distances_itr{it+1}_l{lambda_}_lr{lr}.png')

texture = texture.detach().cpu().numpy()
envmap = envmap.detach().cpu().numpy()
if shading_model == 'diffuse':
    save_img(texture, f'{outdir}/optimized/texture_kd_itr{it+1}_l{lambda_}_lr{lr}.exr', mat_res)
elif shading_model == 'specular':
    save_img(texture[...,:1], f'{outdir}/optimized/texture_r_itr{it+1}_l{lambda_}_lr{lr}.exr', mat_res)
    save_img(texture[...,1:4], f'{outdir}/optimized/texture_eta_itr{it+1}_l{lambda_}_lr{lr}.exr', mat_res)
    save_img(texture[...,4:7], f'{outdir}/optimized/texture_k_itr{it+1}_l{lambda_}_lr{lr}.exr', mat_res)
    save_img(texture[...,7:], f'{outdir}/optimized/texture_ks_itr{it+1}_l{lambda_}_lr{lr}.exr', mat_res)
save_img(envmap, f'{outdir}/optimized/envmap_itr{it+1}_l{lambda_}_lr{lr}.exr', env_res)
v = from_differential(M, u)
_, f, uvs, uv_idx = scene.get_mesh(key, return_uv=True)
v = v.detach().cpu().numpy()

# del scene
# torch.cuda.empty_cache()
# ek.cuda_malloc_trim()

# scene = Scene(scene_info['src'])
scene.reload_mesh(key, v, f, uvs, uv_idx)
scene.reload_mat(key, texture.reshape(mat_res[1], mat_res[0], -1))
scene.reload_envmap(envmap.reshape(env_res[1], env_res[0], -1))
scene.set_opts(res, spp=32, sppe=0, sppse=0)
imgs = scene.renderC('shaded')
masks = scene.renderC('mask')
for i in range(n_sensors):
    save_img(imgs[i], f'{outdir}/optimized/itr{it+1}_{sensor_ids[i]}.png', res)
    save_img(masks[i], f'{outdir}/optimized/itr{it+1}_{sensor_ids[i]}_mask.exr', res)
scene.dump(key, f'{outdir}/optimized/optimized_itr{it+1}_l{lambda_}_lr{lr}.obj')

for id in sensor_ids:
    imgs2video(f"{outdir}/train/{id}/train.*.png", f"{outdir}/train/{id}.mp4", 30)
    imgs2video(f"{outdir}/train/{id}/train_mask.*.exr", f"{outdir}/train/{id}_mask.mp4", 30)

'''
Issues:
    - Remeshing loses UVs (causing problems when jointly optimizing geometry and materials)
    - The surface color gets baked into the environment map during the early stage although
    gradually gets separate afterwards
        - Because of the sudden change of shading model?
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
    - Use scheduler during training
        - try to use exponential falloff
    - Implement upsampling and downsampling remesh algorithms
        - remesh and upsample textures periodically during optimization
        - implement instant mesh
    - Implement the albedo smoothness regularizer as in nvdiffrec
    - The current specular material model is not very compatible with the Disney BSDF material model
        - IOR eta and k is not compatible
        - read the Disney BSDF paper
        - Implement the full Disney BRDF parameterized by roughness, metallic, and albedo
        - Implement Schlicks's approximation of the Fresnel term first
        - Multiple importance sample the two lobes and light
            - Look up Mitsuba 2's rough plastic BSDF
'''