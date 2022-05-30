from glob import glob
import os
import pathlib
import time
import xml.etree.ElementTree as ET
import cv2
import imageio.v2 as iio
import xatlas
import torch
from torch.nn import functional
import numpy as np
import enoki as ek
from enoki.cuda_autodiff import Vector3f as Vector3fD
from enoki.cuda import Vector3f as Vector3fC
import nvdiffrast.torch as dr
from geometry.mesh import Mesh, auto_normals
from geometry.dmtet import DMTetGeometry
from geometry.obj import write_obj
from geometry.util import remesh
from render.scene import Scene
from render.mlptexture import MLPTexture3D
from render.util import wrap

def linear2srgb(img):
    img[img <= 0.0031308] *= 12.92
    img[img > 0.0031308] = 1.055*img[img > 0.0031308]**(1./2.4) - 0.055
    return np.clip(img, 0., 1.)
def srgb2linear(img):
    img[img <= 0.04045] /= 12.92
    img[img > 0.04045] = ((img[img > 0.04045]+0.055)/1.055)**2.4
    return np.clip(img, 0., 1.)

def exr2png(fname):
    img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = linear2srgb(img)
    img = np.uint16(img*65535)
    cv2.imwrite(str(pathlib.Path(fname).with_suffix('.png')), img)

def imgs2video(pattern, dst, fps):
    ext = pathlib.Path(pattern).suffix
    fnames = np.sort(glob(pattern))
    w = iio.get_writer(dst, format='FFMPEG', mode='I', fps=fps)
    if ext == '.exr':
        for fname in fnames:
            img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = linear2srgb(img)
            img *= 255
            img = img.astype(np.uint8)
            w.append_data(img)
    else:
        for fname in fnames:
            w.append_data(iio.imread(fname))
    w.close()

def load_img(fname):
    img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint8:
        img = srgb2linear(img.astype(np.float32)/255)
    if img.dtype == np.uint16:
        img = srgb2linear(img.astype(np.float32)/65535)
    return img
def save_img(img, fname, res: tuple):
    '''
    Save an image of linear color values to `fname`
    '''
    if type(img) == Vector3fD or type(img) == Vector3fC:
        img = img.numpy()
    elif type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()

    img = img.reshape((res[1], res[0], img.shape[-1]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pathlib.Path(fname).suffix != '.exr':
        img = linear2srgb(img)
        img = np.uint16(img*65535)
    cv2.imwrite(fname, img)

def xmlfile2str(fname):
    tree = ET.parse(fname)
    return ET.tostring(tree.getroot(), encoding='unicode')

class NotFoundError(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)
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
        raise NotFoundError('No saved tensors matching the specified number of iteration')

def preprocess_scene(fname):
    def remove_elem(parent, tag):
        for elem in parent.findall(tag):
            parent.remove(elem)
        for elem in parent:
            remove_elem(elem, tag)
    def find_id(parent, id):
        for elem in parent:
            if elem.get('id') == id:
                return (parent, elem)
        for elem in parent:
            a = find_id(elem, id)
            if a is not None:
                return a
    def replace_filename(elem, fname):
        for string in elem.findall('string'):
            if string.get('name') == 'filename':
                old_fname = string.attrib['value']
                string.attrib['value'] = fname
                return old_fname
    def num_of(parent, tag):
        return len(parent.findall(tag))

    '''
    Construct source scene
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'target')
    p[0].remove(p[1])
    p = find_id(root, 'ref_mat')
    p[0].remove(p[1])
    source_scene = ET.tostring(root, encoding='unicode')

    '''
    Construct target scene
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'source')
    p[0].remove(p[1])
    p = find_id(root, 'opt_mat')
    p[0].remove(p[1])
    target_scene = ET.tostring(root, encoding='unicode')

    return {
        'src': source_scene,
        'tgt': target_scene,
        'n_sensors': num_of(root, 'sensor')
    }

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

def renderC_img(xml, integrator, sensor_ids = None, res = (256, 256), spp = 32, load_string=True):
    if load_string == False: xml = xmlfile2str(xml)
    scene = Scene(xml)
    scene.set_opts(res, spp, sppe=0, sppse=0)
    imgs = scene.renderC(integrator, sensor_ids)
    assert(imgs is not None)
    return scene, imgs

def prepare_for_mesh_opt(ckp_path):
    ckp = torch.load(ckp_path)
    dmtet = DMTetGeometry(64, 4, ckp['sdf'], ckp['deform'])
    kd_min, kd_max = [0., 0., 0.], [1., 1., 1.]
    material = MLPTexture3D(dmtet.getAABB(),
                min_max=torch.stack([torch.tensor(kd_min, device='cuda'), torch.tensor(kd_max, device='cuda')]))
    material.load_state_dict(ckp['mat'])
    material.eval()
    
    m = dmtet.getMesh()
    m.v_pos, m.t_pos_idx = remesh(m.v_pos, m.t_pos_idx)
    m = extract_texture(m, material)
    
    write_obj('data/meshes/source.obj', auto_normals(m), True)

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
    tex = texture_3d.sample(pos[0])
    # del glctx
    
    return Mesh(v_tex=uvs, t_tex_idx=uv_idx, material=tex, base=mesh)

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