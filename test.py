import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import potpourri3d as pp3d
from enoki.cuda_autodiff import Int32 as IntD, Float32 as FloatD, Vector3m as Vector3mD, Vector2f as Vector2fD, Vector3f as Vector3fD, Vector4f as Vector4fD, Matrix4f as Matrix4fD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC, Matrix4f as Matrix4fC
import psdr_cuda
from torch.utils.data import DataLoader
import potpourri3d as pp3d
import polyscope as ps
from torch.nn import functional
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from geometry.obj import load_obj, write_obj
from xml_util import keep_sensors, preprocess_scene, xmlfile2str
from util import *
from geometry.dmtet import *
from geometry.util import remesh, transform_ek
from render.util import gen_tex, sample, scale_img, wrap, flip_x, flip_y
import denoise.oidn as oidn
import denoise.svgf as svgf

def test_mem_capacity(res, spp):
    integrator = psdr_cuda.DirectIntegrator(1, 1)
    scene_info = preprocess_scene('example.xml')
    scene = Scene(scene_info['tgt'])
    scene.set_opts(res, spp, sppe=0, sppse=0)
    scene.prepare()
    for _ in range(100):
        img = integrator.renderC(scene._scene, 2)
    save_img(img, 'img.exr', res)

def display_meshes(fnames):
    ps.init()
    for i, fname in enumerate(fnames):
        v, f = pp3d.read_mesh(fname)
        ps.register_surface_mesh(f'm{i}', v, f, False)
    ps.show()

def test_mlp_tex():
    dmtet = DMTetGeometry(128, 1)
    kd_min, kd_max = [0., 0., 0.], [1., 1., 1.]
    mat = MLPTexture3D(dmtet.getAABB(),
        min_max=torch.stack([torch.tensor(kd_min, device='cuda'), torch.tensor(kd_max, device='cuda')]))
    m = dmtet.getMesh()
    samples = mat.sample(m.v_pos)
    samples = samples[m.t_pos_idx].reshape(-1,3)
    uvs = m.v_tex[m.t_tex_idx].reshape(-1,2)
    res = 128
    tex = gen_tex(samples, uvs, res)
    m.material = tex
    write_obj('sdf.obj', mesh.auto_normals(dmtet.getMesh()), save_material=True)

    scene_info = preprocess_scene('example.xml', sdf_path='sdf.obj')

    scene = Scene(scene_info['sdf'])
    scene.set_opts((320, 180), 64, 3, sppe=0, sppse=0)
    integrator = psdr_cuda.DirectIntegrator()
    img = scene.renderC(integrator, [2])
    save_img(img[0], 'img.exr', (320, 180))

def test_gen_tex():
    scene_info = preprocess_scene('example.xml')
    scene = Scene(scene_info['src'])
    scene.set_opts((320, 180), 64, 3, sppe=0, sppse=0)

    m = scene._scene.param_map['Mesh[0]']
    v = m.vertex_positions.torch()
    f = m.face_indices.torch()
    uvs = m.vertex_uv.torch()
    uv_idx = m.face_uv_indices.torch()
    tex = m.bsdf.reflectance.data.torch()
    res = m.bsdf.reflectance.resolution

    tex = tex.reshape(*res, 3)
    uvs = uvs[uv_idx.to(torch.long)].reshape(-1,2)
    uvs = wrap(flip_y(uvs))
    samples = sample(tex, uvs, 'bilinear')
    tex = gen_tex(samples, uvs, 16)
    save_img(tex, 'tex.exr', (16, 16))

def test_vert_color():
    scene_info = preprocess_scene('example.xml')
    scene = Scene(scene_info['src'])
    scene.set_opts((320, 180), 64, 3, sppe=0, sppse=0)
    img = scene.renderC(psdr_cuda.DirectIntegrator(), [2])
    save_img(img[0], 'img.exr', (320, 180))

    m = scene._scene.param_map['Mesh[0]']
    v = m.vertex_positions.torch()
    f = m.face_indices.numpy()
    uvs = wrap(flip_y(m.vertex_uv.torch()))
    uv_idx = m.face_uv_indices.numpy().flatten()
    tex = psdr_cuda.Bitmap3fD('data/meshes/spot/spot_texture.exr')

    vert_idx = f.flatten()
    assert(vert_idx.shape[0] == uv_idx.shape[0])
    vert_idx_unique, index = np.unique(vert_idx, return_index=True)
    assert(vert_idx_unique.shape[0] == v.shape[0])
    uv_idx = torch.from_numpy(uv_idx[index]).cuda().long()
    vert_idx_unique = torch.from_numpy(vert_idx_unique).cuda().long()

    samples = sample(tex.data.torch().reshape(1024,1024,3), uvs[uv_idx], 'bilinear')
    vert_colors = torch.zeros_like(v).scatter(0, vert_idx_unique.reshape(-1,1).expand(-1,3), samples)
    # vert_colors = psdr_cuda.Bitmap3fD(1, vert_colors.shape[0], Vector3fD(vert_colors))
    # m.bsdf.reflectance = vert_colors

    scene.set_opts((320, 180), 64, 3, sppe=0, sppse=0)
    scene.reload_mat('Mesh[0]', vert_colors)
    img = scene.renderC(psdr_cuda.DirectIntegrator(), [2])
    save_img(img[0], 'img1.exr', (320, 180))

def test_demod():
    scene_info = preprocess_scene('data/scenes/spot_env/spot_env.xml')
    scene = Scene(scene_info['tgt'])
    res = (1280, 720)
    scene.set_opts(res, spp=8)

    img_demod0 = scene.renderC([0], 'demodulated')[0]
    img1 = scene.renderC([0], 'shaded')[0]
    img_mask = scene.renderC([0], 'mask')[0]
    img_mask = np.broadcast_to(img_mask, [*img_mask.shape[:-1], 3])
    img_alb = scene.renderC([0], 'albedo')[0]
    mask = img_mask > 0.

    img0 = np.copy(img_demod0)
    img0[mask] = img0[mask] * img_alb[mask]
    img_demod1 = np.copy(img1)
    img_demod1[mask] = img_demod1[mask] / (img_alb[mask] + 1e-5)

    denoiser = oidn.load_denoiser('hdr')
    img_denoised = denoiser(torch.from_numpy(img_demod0.reshape(1, res[1], res[0], 3)).cuda())
    
    save_img(img_demod0, 'img_demod0.exr', res)
    save_img(img0, 'img0.exr', res)
    save_img(img1, 'img1.exr', res)
    save_img(img_demod1, 'img_demod1.exr', res)
    save_img(img_denoised[0], 'img_denoised.exr')

def test_deriv(img):
    device = img.device
    img = img[None, ...]
    shape = img.shape
    res = (shape[2], shape[1])
    img = torch.permute(img, (0, 3, 1, 2))

    v, u = torch.meshgrid([torch.arange(0, res[1], dtype=torch.float32, device=device),
                torch.arange(0, res[0], dtype=torch.float32, device=device)], indexing='ij')
    u = u + 0.5
    v = v + 0.5
    h = 0.1

    u0 = u - h
    v0 = v - h
    u1 = u + h
    v1 = v + h
    u = u / res[0]
    u0 = u0 / res[0]
    u1 = u1 / res[0]
    v0 = v0 / res[1]
    v = v / res[1]
    v1 = v1 / res[1]

    u0v = torch.stack([u0, v], dim=-1)
    u1v = torch.stack([u1, v], dim=-1)
    uv0 = torch.stack([u, v0], dim=-1)
    uv1 = torch.stack([u, v1], dim=-1)
    u0v = u0v.expand(shape[0], res[1], res[0], 2)
    u1v = u1v.expand(shape[0], res[1], res[0], 2)
    uv0 = uv0.expand(shape[0], res[1], res[0], 2)
    uv1 = uv1.expand(shape[0], res[1], res[0], 2)

    img_u0 = functional.grid_sample(img, u0v*2 - 1, mode='bilinear', align_corners=False)
    img_u1 = functional.grid_sample(img, u1v*2 - 1, mode='bilinear', align_corners=False)
    img_v0 = functional.grid_sample(img, uv0*2 - 1, mode='bilinear', align_corners=False)
    img_v1 = functional.grid_sample(img, uv1*2 - 1, mode='bilinear', align_corners=False)

    didu = (img_u1 - img_u0) / 2 / h
    didv = (img_v1 - img_v0) / 2 / h

    didu = torch.permute(didu, (0, 2, 3, 1))
    didv = torch.permute(didv, (0, 2, 3, 1))
    img_u0 = torch.permute(img_u0, (0, 2, 3, 1))
    img_u1 = torch.permute(img_u1, (0, 2, 3, 1))

    didu = torch.sigmoid(didu)
    didu = torch.abs(didu - 0.5) * 2
    didv = torch.sigmoid(didv)
    didv = torch.abs(didv - 0.5) * 2

    save_img(didu[0], 'didu.exr')
    save_img(didv[0], 'didv.exr')
    save_img(img_u0[0], 'u0.exr')
    save_img(img_u1[0], 'u1.exr')

def test_deriv1(img):
    pad = torch.nn.ReplicationPad2d(1)
    device = img.device
    img = img[None, ...]
    shape = img.shape
    res = (shape[2], shape[1])
    img = torch.permute(img, (0, 3, 1, 2))

    img_padded = pad(img)
    img_u0 = img_padded[:, :, 1:-1, 0:-2]
    img_u1 = img_padded[:, :, 1:-1, 2:]
    img_v0 = img_padded[:, :, 0:-2, 1:-1]
    img_v1 = img_padded[:, :, 2:, 1:-1]

    didu = (img_u1 - img_u0) / 2
    didv = (img_v1 - img_v0) / 2

    didu = torch.permute(didu, (0, 2, 3, 1))
    didv = torch.permute(didv, (0, 2, 3, 1))
    img_u0 = torch.permute(img_u0, (0, 2, 3, 1))
    img_u1 = torch.permute(img_u1, (0, 2, 3, 1))

    didu = torch.sigmoid(didu)
    didu = torch.abs(didu - 0.5) * 2
    didv = torch.sigmoid(didv)
    didv = torch.abs(didv - 0.5) * 2

    save_img(didu[0], 'didu_1.exr')
    save_img(didv[0], 'didv_1.exr')
    save_img(img_u0[0], 'u0_1.exr')
    save_img(img_u1[0], 'u1_1.exr')

def test_SVGF(size):
    scene_info = preprocess_scene('data/scenes/spot_env/spot_env.xml')
    scene = Scene(scene_info['tgt'])
    res = (1280, 720)
    scene.set_opts(res, spp=1)
    img = scene.renderC([0], 'shaded')[0]
    img_depth = scene.renderC([0], 'depth')[0]
    img_nrm = scene.renderC([0], 'normal')[0]

    img_depth = np.mean(img_depth, axis=-1, keepdims=True)
    img = torch.from_numpy(img)
    img_depth = torch.from_numpy(img_depth)
    img_nrm = torch.from_numpy(img_nrm)
    img = img.reshape(1, res[1], res[0], -1)
    img_depth = img_depth.reshape(1, res[1], res[0], -1)
    img_nrm = img_nrm.reshape(1, res[1], res[0], -1)

    save_img(img[0], 'img.exr')
    save_img(img_depth[0], 'depth.exr')
    save_img(img_nrm[0], 'normal.exr')

    denoiser = svgf.load_denoiser()
    denoised = denoiser(img, img_depth, img_nrm, size, 2, 1, 128)

    save_img(denoised[0], f'denoised_{size}.exr')


filenames = next(os.walk('misc'))[2]
for filename in filenames:
    if filename.endswith('.exr'):
        print(filename)
        exr2png(os.path.join('misc', filename))

# prepare_for_mesh_opt('output/drums_dmtet/optimized/ckp.1000.tar', 64, 2.5, 'diffuse')

# dump_opt_mesh('output/drums/optimized/ckp.1000.tar', './')

# preprocess_nerf_synthetic('data/nerf_synthetic/drums/transforms_train.json',
#                         'data/scenes/nerf_synthetic/drums.xml')

# img = load_img('data/meshes/spot/spot_texture.exr')
# img = torch.from_numpy(img)
# test_deriv(img)

# v, u = torch.meshgrid([torch.arange(0, 4, dtype=torch.float32),
#             torch.arange(0, 3, dtype=torch.float32)], indexing='ij')
# print(u)
# print(v)
# u = u + 0.5
# v = v + 0.5
# uv = torch.stack([u, v], dim=-1)
# print(uv)

# prepare_for_mesh_opt('output/lego_dmtet/optimized/ckp.1000.tar', 64, 2.5)
# preprocess_nerf_synthetic('data/nerf_synthetic/drums/transforms_train.json',
#                             'data/scenes/nerf_synthetic/drums.xml')

# v, _ = scene.get_mesh('Mesh[0]')
# v = torch.from_numpy(v).cuda().requires_grad_()

# v = Vector3fD(v)
# P = FloatD(0.)
# ek.set_requires_gradient(P)

# scene._scene.param_map['Mesh[0]'].vertex_positions = v
# scene._scene.param_map['Mesh[0]'].set_transform(Matrix4fD.translate(Vector3fD(0., 1., 0.)*P))
# scene._scene.configure()

# img = scene._integrator.renderD(scene._scene, 0)

# ek.forward(P)
# img_grad = ek.gradient(img)

# save_img(img, 'img.exr', res)
# save_img(img_grad, 'img_grad.exr', res)

# mat = scene.get_mat('Mesh[0]')
# mat = torch.from_numpy(mat.reshape(-1,3)).cuda().requires_grad_()

# envmap = scene.get_envmap()
# envmap = torch.from_numpy(envmap.reshape(-1,3)).cuda().requires_grad_()

# img_demod, mask, img_alb = scene.renderD_demod(v, mat, envmap, 'Mesh[0]', [0])
# img, mask = scene.renderD(v, mat, envmap, 'Mesh[0]', [0])


# save_img(img[0], 'img.exr', res)
# save_img(img_demod[0], 'img_demod.exr', res)
# save_img(mask[0], 'mask.exr', res)
# save_img(img_alb[0], 'alb.exr', res)

# loss = torch.sum(img_demod) + torch.sum(mask) + torch.sum(img_alb)
# loss = torch.sum(img) + torch.sum(mask)

# loss.backward()

# img_nrm = scene.renderC([0], 'normal')
# save_img(img_nrm[0], 'nrm.exr', res)
# img_nrm = torch.from_numpy(img_nrm[0]).cuda().requires_grad_()
# img_nrm = img_nrm.unsqueeze(0)

# denoiser = load_denoiser('hdr_alb_nrm')
# img_denoised = denoiser(
#                 torch.cat([img_demod, img_alb, img_nrm], dim=-1).reshape(1, res[1], res[0], -1))
# save_img(img_denoised[0], 'img_denoised_hdr_alb_nrm.exr')

# imgs = scene.renderC([0], img_type='shaded')
# imgs_normal = scene.renderC([0], img_type='normal')
# imgs_alb = scene.renderC([0], img_type='albedo')
# imgs_mask = scene.renderC(integrator_mask, [0])

# for i in range(len(imgs)):
#     save_img(imgs[i], f'img{i}.exr', res)
#     save_img(imgs_normal[i], f'img{i}_normal.exr', res)
    # save_img(imgs_uv[i], f'img{i}_uv.exr', res)
    # save_img(imgs_mask[i], f'img{i}_mask.exr', res)
    # mask = Vector3fD(imgs_mask[i]) > 0.
    # uvs = Vector2fD(imgs_uv[i][:,:2])
    # img_albedo = albedo.eval(uvs)
    # img_albedo = ek.select(mask, img_albedo, 0.)
    # save_img(imgs_alb[i], f'img{i}_albedo.exr', res)

# denoiser = load_denoiser('hdr_alb_nrm')

# img = load_img('img0.exr')
# alb = load_img('img0_albedo.exr')
# nrm = load_img('img0_normal.exr')
# img = scale_img(img, (540, 960)).cuda()
# alb = scale_img(alb, (540, 960)).cuda()
# nrm = scale_img(nrm, (540, 960)).cuda()

# torch.autograd.set_detect_anomaly(True)

# res = img.shape[:2]
# res = (res[1], res[0])
# input = torch.cat([img, alb, nrm], dim=-1)
# input = input[None,...]
# input.requires_grad_()

# output = denoiser(input)
# loss = output.mean()
# loss.backward()
# print(input.grad.mean())
# save_img(output[0], 'denoised.exr', res)

# _x = FloatD()

# class EnokiSquare(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         assert(x.requires_grad)
#         global _x
#         ctx.x = FloatD(x)

#         ek.set_requires_gradient(ctx.x)

#         _x = ctx.x

#         ctx.out = _x * _x

#         return ctx.out.torch()

#     @staticmethod
#     def backward(ctx, grad_out):
#         ek.set_gradient(ctx.out, FloatC(grad_out))

#         FloatD.backward()

#         grad_x = ek.gradient(ctx.x).torch()

#         del ctx.x, ctx.out

#         return grad_x
# ek_sqr = EnokiSquare.apply

# class EnokiCube(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         assert(x.requires_grad)
#         global _x
#         ctx.x = FloatD(x)

#         ek.set_requires_gradient(ctx.x)

#         _x = ctx.x

#         ctx.out = _x * _x * _x

#         return ctx.out.torch()

#     @staticmethod
#     def backward(ctx, grad_out):
#         ek.set_gradient(ctx.out, FloatC(grad_out))

#         FloatD.backward()

#         grad_x = ek.gradient(ctx.x).torch()

#         del ctx.x, ctx.out

#         return grad_x
# ek_cube = EnokiCube.apply

# x = torch.arange(1, 11, dtype=torch.float32, device='cuda', requires_grad=True)
# print(x)

# x_2 = ek_sqr(x)
# print(x_2)
# x_3 = ek_cube(x)
# print(x_3)

# loss = torch.sum(x_2)
# loss += torch.sum(x_3)

# loss.backward()

# print(x.grad)

# input()