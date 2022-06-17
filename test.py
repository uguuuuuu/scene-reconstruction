import os

from zmq import device

from denoise.oidn.load_denoiser import load_denoiser
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
from geometry.util import remesh
from render.util import gen_tex, sample, wrap, flip_x, flip_y

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


# prepare_for_mesh_opt('output/lego_dmtet/optimized/ckp.1000.tar', 64, 2.5)
# preprocess_nerf_synthetic('data/nerf_synthetic/drums/transforms_train.json',
#                             'data/scenes/nerf_synthetic/drums.xml')

scene_info = preprocess_scene('data/scenes/spot_env/spot_env.xml')
scene = Scene(scene_info['tgt'])
# albedo = scene._scene.param_map['Mesh[0]'].bsdf.reflectance
res = (1920, 1080)
scene.set_opts(res, spp=1)
integrator = psdr_cuda.DirectIntegrator()
# integrator_normal = psdr_cuda.FieldExtractionIntegrator('shNormal')
# integrator_uv = psdr_cuda.FieldExtractionIntegrator('uv')
# integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')

# imgs = scene.renderC(integrator, [0])
# imgs_normal = scene.renderC(integrator_normal, [0])
# imgs_uv = scene.renderC(integrator_uv, [0])
# imgs_mask = scene.renderC(integrator_mask, [0])

# for i in range(len(imgs)):
#     save_img(imgs[i], f'img{i}.exr', res)
#     save_img(imgs_normal[i], f'img{i}_normal.exr', res)
#     save_img(imgs_uv[i], f'img{i}_uv.exr', res)
#     save_img(imgs_mask[i], f'img{i}_mask.exr', res)
#     mask = Vector3fD(imgs_mask[i]) > 0.
#     uvs = Vector2fD(imgs_uv[i][:,:2])
#     img_albedo = albedo.eval(uvs)
#     img_albedo = ek.select(mask, img_albedo, 0.)
#     save_img(img_albedo, f'img{i}_albedo.exr', res)

v, f = scene.get_mesh('Mesh[0]')
print(v.shape)
scene.reload_mesh('Mesh[0]', v, f)
mat = np.zeros_like(v, dtype=np.float32)
mat[...,0] = 1.
print(mat.shape)
scene.reload_mat('Mesh[0]', mat)
imgs = scene.renderC(integrator, [0])
save_img(imgs[0], 'img1.exr', res)

# denoiser = load_denoiser('hdr_alb_nrm')

# img = load_img('img0.exr')
# alb = load_img('img0_albedo.exr')
# nrm = load_img('img0_normal.exr')

# res = img.shape[:2]
# res = (res[1], res[0])
# img = torch.from_numpy(img).cuda()
# alb = torch.from_numpy(alb).cuda()
# nrm = torch.from_numpy(nrm).cuda()
# input = torch.cat([img, alb, nrm], dim=-1)

# output = denoiser(input)
# save_img(output, 'denoised.exr', res)