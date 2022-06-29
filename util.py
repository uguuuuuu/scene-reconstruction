from glob import glob
import os
import pathlib
import time
import xml.etree.ElementTree as ET
import json
import cv2
import imageio.v2 as iio
import xatlas
import torch
from torch.nn import functional
import numpy as np
import enoki as ek
from enoki.cuda_autodiff import Vector3f as Vector3fD, Float32 as FloatD
from enoki.cuda import Vector3f as Vector3fC, Float32 as FloatC
import nvdiffrast.torch as dr
from geometry.mesh import Mesh, auto_normals
from geometry.dmtet import DMTetGeometry
from geometry.obj import write_obj
from geometry.util import remesh
from render.scene import Scene
from render.mlptexture import MLPTexture3D
from render.util import wrap
from xml_util import formatxml, xmlfile2str

def linear2srgb(img):
    img[img <= 0.0031308] *= 12.92
    img[img > 0.0031308] = 1.055*img[img > 0.0031308]**(1./2.4) - 0.055
    return np.clip(img, 0., 1.)
def srgb2linear(img):
    img[img <= 0.04045] /= 12.92
    img[img > 0.04045] = ((img[img > 0.04045]+0.055)/1.055)**2.4
    return np.clip(img, 0., 1.)

def exr2png(fname):
    img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    img = linear2srgb(img)
    img = np.uint16(img*65535)
    cv2.imwrite(str(pathlib.Path(fname).with_suffix('.png')), img)

def imgs2video(pattern, dst, fps):
    fnames = np.sort(glob(pattern))
    w = iio.get_writer(dst, format='FFMPEG', mode='I', fps=fps)

    for fname in fnames:
        img = load_img(fname)
        img = linear2srgb(img)
        img *= 255
        img = img.astype(np.uint8)
        w.append_data(img)
    w.close()

def load_img(fname):
    img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    n_channels = img.shape[-1]
    if n_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif n_channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        n_channels = 1
        img = img[...,None]

    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255
        img = srgb2linear(img)
    elif img.dtype == np.uint16:
        img = img.astype(np.float32)/65535
        img = srgb2linear(img)

    return img
def save_img(img, fname, res=None):
    '''
    Save an image of linear color values to `fname`
    '''
    t = type(img)
    if t == Vector3fD or t == Vector3fC:
        img = img.numpy()
    if t == FloatC or t == FloatD:
        img = img.numpy()
        img = img[...,None]
    elif t == torch.Tensor:
        img = img.detach().cpu().numpy()

    if res is None:
        assert(img.ndim == 3)
        res = img.shape[:2]
        res = (res[1], res[0])

    n_channels = img.shape[-1]
    if n_channels != 1 and n_channels !=3 and n_channels !=4:
        assert(img.ndim == 1)
        n_channels = 1
    img = img.reshape((res[1], res[0], n_channels))
    if n_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif n_channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    if pathlib.Path(fname).suffix != '.exr':
        img = linear2srgb(img)
        img = np.uint16(img*65535)
    cv2.imwrite(fname, img)

def load_ckp(pattern, n_itr = -1):
    '''
    Load a PyTorch checkpoint

    Parameters
    ----------
    pattern: str
        Pattern of the tensor to load
    n_itr: None or int
        load the latest if -1, otherwise load the tensor with the specified number of iteration
    
    Returns
    ---------
    checkpoint:
        The loaded checkpoint
    n_itr:
        The iteration at which the checkpoint was saved
    '''
    paths = glob(pattern)
    if len(paths) == 0: return None, 0
    paths = [pathlib.Path(p) for p in paths]

    if n_itr == -1:
        path, itr = paths[0], int(paths[0].suffixes[-2][1:])
        for p in paths:
            itr = int(path.suffixes[-2][1:])
            j = int(p.suffixes[-2][1:])
            if j > itr: path, itr = p, j
        return torch.load(path), itr
    else:
        for p in paths:
            if int(p.suffixes[-2][1:]) == n_itr:
                return torch.load(p), n_itr
        raise FileNotFoundError('No saved tensors matching the specified number of iteration')

def unique(x: torch.Tensor, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

def lerp(a, b, t):
    return a * (1 - t) + b * t

def lgt_reg_loss(lgt, ref):
    def linear2srgb(f):
        return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)
    def tonemap(x):
        return linear2srgb(torch.log(torch.clamp(x, min=0, max=65535) + 1))

    return functional.l1_loss(torch.mean(tonemap(lgt), dim=-1), torch.max(tonemap(ref), dim=-1)[0])

def renderC_img(xml, sensor_ids = None, res = (256, 256), spp = 32, load_string=True, img_type='shaded'):
    if load_string == False: xml = xmlfile2str(xml)
    scene = Scene(xml)
    scene.set_opts(res, spp, sppe=0, sppse=0)
    imgs = scene.renderC(sensor_ids, img_type)
    assert(imgs is not None)
    return scene, imgs

def prepare_for_mesh_opt(ckp_path, tet_res, tet_scale, shading_model='diffuse'):
    ckp = torch.load(ckp_path)
    dmtet = DMTetGeometry(tet_res, tet_scale, ckp['sdf'], ckp['deform'])

    if shading_model == 'diffuse':
        mat_min = torch.zeros(3, dtype=torch.float32, device='cuda')
        mat_max = torch.ones(3, dtype=torch.float32, device='cuda')
    else:
        mat_min = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda')
        mat_max = torch.tensor([1., 5., 5., 5., 5., 5., 5., 1., 1., 1.], device='cuda')

    material = MLPTexture3D(dmtet.getAABB(), min_max=torch.stack([mat_min, mat_max]))
    material.load_state_dict(ckp['mat'])
    material.eval()
    
    m = dmtet.getMesh()
    m.v_pos, m.t_pos_idx = remesh(m.v_pos, m.t_pos_idx)
    m = extract_texture(m, material)
    
    write_obj('data/meshes/source.obj', auto_normals(m), True)

    envmap = ckp['envmap']
    env_res = ckp['env_res']
    save_img(envmap, 'data/envmaps/source_envmap.exr', env_res)

    sensor_ids = ckp['sensor_ids']
    sensor_ids = np.array(sensor_ids)
    np.save('data/sensor_ids.npy', sensor_ids)

def preprocess_nerf_synthetic(cfg_path, save_path):
    folder = os.path.dirname(cfg_path)
    cfg = json.load(open(cfg_path, 'r'))
    root = ET.Element('scene')
    tree = ET.ElementTree(root)

    img_path = os.path.join(folder, cfg['frames'][0]['file_path'])+'.png'
    img = load_img(img_path)
    res = img.shape[:2]
    res = (res[1], res[0])

    fovx = np.rad2deg(cfg['camera_angle_x'])

    root.append(ET.Comment('Cameras'))
    for i in range(len(cfg['frames'])):
        sensor = ET.SubElement(root, 'sensor', {'type': 'perspective'})
        ET.SubElement(sensor, 'string', {'name':'fov_axis', 'value':'x'})
        ET.SubElement(sensor, 'float', {'name':'fov', 'value': f'{fovx}'})
        ET.SubElement(sensor, 'float', {'name':'near_clip', 'value': '0.1'})
        ET.SubElement(sensor, 'float', {'name':'far_clip', 'value': '1000'})

        transform = ET.SubElement(sensor, 'transform', {'name':'to_world'})
        mat = np.array(cfg['frames'][i]['transform_matrix'])
        mat = mat @ rotate_y(np.deg2rad(180)).numpy()
        mat = mat.flatten()
        mat_str = ''
        for e in mat:
            mat_str += str(e)+' '
        ET.SubElement(transform, 'matrix', {'value':mat_str})

        if i == 0:
            sampler = ET.SubElement(sensor, 'sampler', {'type':'independent'})
            ET.SubElement(sampler, 'integer', {'name':'sample_count', 'value':'1'})
            film = ET.SubElement(sensor, 'film', {'type':'hdrfilm'})
            ET.SubElement(film, 'integer', {'name':'width', 'value':f'{res[0]}'})
            ET.SubElement(film, 'integer', {'name':'height', 'value':f'{res[1]}'})

    root.append(ET.Comment('Materials'))
    bsdf = ET.SubElement(root, 'bsdf', {'type':'diffuse', 'id':'opt_mat'})
    ET.SubElement(bsdf, 'rgb', {'name':'reflectance', 'value':'1, 0, 0'})

    root.append(ET.Comment('Emitters'))
    envmap = ET.SubElement(root, 'emitter', {'type':'envmap'})
    ET.SubElement(envmap, 'string', {'name':'filename', 'value':'./data/envmaps/kloppenheim_06_2k.exr'})

    root.append(ET.Comment('Shapes'))
    shape = ET.SubElement(root, 'shape', {'type':'obj', 'id':'source'})
    ET.SubElement(shape, 'string', {'name':'filename', 'value':'./data/meshes/source.obj'})
    ET.SubElement(shape, 'ref', {'id':'opt_mat'})

    formatxml(root)
    
    tree.write(save_path)

class TimerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args) 
class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError(f'Timer is running. Use stop() to stop it')
        self._start_time = time.perf_counter()
    
    def stop(self):
        if self._start_time is None:
            raise TimerError(f'Timer is not running. Use .start() to start it')
        elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed

'''
Adapted from https://github.com/NVlabs/nvdiffrec
'''
def lr_schedule(iter, warmup_iter):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.001)) # Exponential falloff from [1.0, 0.1] over 1k epochs.
def extract_texture(mesh, texture_3d, res=(1024,1024)):
    v = mesh.v_pos.detach().cpu().numpy()
    f = mesh.t_pos_idx.detach().cpu().numpy()
    _, uv_idx, uvs = xatlas.parametrize(v, f)

    uvs = torch.from_numpy(uvs).cuda()
    uv_idx = torch.from_numpy(uv_idx.astype(np.int32)).cuda()
    # Transform texture coordinates into NDC
    uvs_ = wrap(uvs)*2-1
    # Add a batch dimension
    uvs_ = uvs_[None,...]
    # To clip space for rasterization
    uvs_ = torch.cat([uvs_, torch.zeros_like(uvs_[...,0:1]), torch.ones_like(uvs_[...,0:1])], dim=-1)
    glctx = dr.RasterizeGLContext()
    rast, _ = dr.rasterize(glctx, uvs_, uv_idx, res)
    pos, _ = dr.interpolate(mesh.v_pos.contiguous(), rast, mesh.t_pos_idx.int())
    print(pos.shape)
    tex = texture_3d.sample(pos[0])
    # del glctx
    
    return Mesh(v_tex=uvs, t_tex_idx=uv_idx, material=tex, base=mesh)
def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)