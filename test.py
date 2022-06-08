import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import potpourri3d as pp3d
from enoki.cuda_autodiff import Int32 as IntD, Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD, Matrix4f as Matrix4fD
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


prepare_for_mesh_opt('output/lego_dmtet/optimized/ckp.1000.tar', 64, 2.5)
# preprocess_nerf_synthetic('data/nerf_synthetic/lego/transforms_train.json',
#                             'data/scenes/nerf_synthetic/lego.xml')