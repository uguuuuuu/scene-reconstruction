from glob import glob
import pathlib
import time
import xml.etree.ElementTree as ET
import cv2
import imageio.v2 as iio
import torch
import numpy as np
import psdr_cuda
from enoki.cuda_autodiff import Vector3f as Vector3fD, Float32 as FloatD
from enoki.cuda import Vector3f as Vector3fC
from pyremesh import remesh_botsch
from geometry.mesh import Mesh, auto_normals

def linear2srgb(img):
    img[img <= 0.0031308] *= 12.92
    img[img > 0.0031308] = 1.055*img[img > 0.0031308]**(1./2.4) - 0.055
    img = np.uint16(np.clip(img, 0., 1.)*65535)
    return img
def srgb2linear(img):
    img[img <= 0.04045] /= 12.92
    img[img > 0.04045] = ((img[img > 0.04045]+0.055)/1.055)**2.4
    return img
def exr2png(fname):
    img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = linear2srgb(img)
    cv2.imwrite(str(pathlib.Path(fname).with_suffix('.png')), img)
def imgs2video(pattern, dst, fps):
    ext = pathlib.Path(pattern).suffix
    fnames = np.sort(glob(pattern))
    w = iio.get_writer(dst, format='FFMPEG', mode='I', fps=fps)
    if ext == '.exr':
        for fname in fnames:
            img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = linear2srgb(img).astype(np.uint8)
            w.append_data(img)
    else:
        for fname in fnames:
            w.append_data(iio.imread(fname))
    w.close()

def save_img(img, fname, res: tuple):
    if type(img) == Vector3fD or type(img) == Vector3fC:
        i = img.numpy()
    elif type(img) == torch.Tensor:
        i = img.detach().cpu().numpy()

    i = i.reshape((res[1], res[0], i.shape[-1]))
    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
    if pathlib.Path(fname).suffix != '.exr':
        i = linear2srgb(i)
    cv2.imwrite(fname, i)

class NotFoundError(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)
def load_tensor(pattern, n_itr = -1):
    '''
    Load a PyTorch tensor

    Parameters
    ----------
    pattern: str
        Pattern of the tensor to load
    n_itr: None or int
        load the latest if None, otherwise load the tensor with the specified number of iteration
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

def preprocess_scene(fname, remesh_path=None):
    def remove_elem(parent, tag):
        for elem in parent.findall(tag):
            parent.remove(elem)
        for elem in parent:
            remove_elem(elem, tag)

    tree = ET.parse(fname); root = tree.getroot()
    for shape in root.findall('shape'):
        try:
            if shape.attrib['id'] == 'target':
                root.remove(shape)
            elif shape.attrib['id'] == 'source':
                source_shape = shape
        except KeyError:
            continue
    source_scene = ET.tostring(root, encoding='unicode')

    remesh_scene = None
    if remesh_path is not None:
        for string in source_shape.findall('string'):
            if string.attrib['name'] == 'filename':
                filename_tag = string
                source_filename = string.attrib['value']
                string.attrib['value'] = remesh_path
        remesh_scene = ET.tostring(root, encoding='unicode')
        filename_tag.attrib['value'] = source_filename

    remove_elem(root, 'emitter')
    source_mask = ET.tostring(root, encoding='unicode')

    remesh_mask = None
    if remesh_path is not None:
        filename_tag.attrib['value'] = remesh_path
        remesh_mask = ET.tostring(root, encoding='unicode')

    tree = ET.parse(fname); root = tree.getroot()
    for shape in root.findall('shape'):
        try:
            if shape.attrib['id'] == 'source':
                root.remove(shape)
        except KeyError:
            continue
    target_scene = ET.tostring(root, encoding='unicode')
    remove_elem(root, 'emitter')
    target_mask = ET.tostring(root, encoding='unicode')

    return {
        'src': source_scene,
        'tgt': target_scene,
        'rm': remesh_scene,
        'src_mask': source_mask,
        'tgt_mask': target_mask,
        'rm_mask': remesh_mask
    }
            
def transform(v: torch.Tensor, mat: torch.Tensor):
    v = torch.cat([v, torch.ones([v.shape[0], 1], device='cuda')], dim=-1)
    v = v @ mat.t()
    v = v[:,:3] / v[:,3:]
    return v

def renderC_img(xml, integrator, sensor_ids, res = (256, 256), spp = 32, load_string=True):
    scene = psdr_cuda.Scene()
    scene.load_string(xml, False) if load_string else scene.load_file(xml, False)
    scene.opts.width = res[0]
    scene.opts.height = res[1]
    scene.opts.spp = spp
    scene.opts.log_level = 0
    scene.configure()
    imgs = []
    for i in sensor_ids:
        img = integrator.renderC(scene, sensor_id=i)
        imgs.append(img)
    return scene, imgs

'''
    Adapted from the "Large Steps in Inverse Rendering of Geometry" paper
    https://rgl.epfl.ch/publications/Nicolet2021Large
'''
def laplacian_uniform(V, faces):
    """
    Compute the uniform Laplacian

    Parameters
    ----------
    V : scalar
        Number of vertices.
    faces : torch.Tensor
        array of triangle faces.
    """

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()
def regularization_loss(L: torch.Tensor, v: torch.Tensor, sqr = False):
    '''
        Compute the Laplacian regularization term

        Parameters
        ----------
        L : torch.Tensor
            Uniform Laplacian 
        v : torch.Tensor
            Vertex positions
        sqr: bool
            Whether to square L
    '''
    return (L@v).square().mean() if sqr else (v * (L@v)).mean()
def remove_duplicates(v, f):
    """
    Generate a mesh representation with no duplicates and
    return it along with the mapping to the original mesh layout.
    """

    unique_verts, inverse = torch.unique(v, dim=0, return_inverse=True)
    new_faces = inverse[f.long()]
    return unique_verts, new_faces, inverse

def average_edge_length(verts, faces):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    face_verts = verts[faces.long()]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    return (A + B + C).sum() / faces.shape[0] / 3
'''
'''

@torch.no_grad()
def remesh(v: torch.Tensor, f: torch.Tensor):
    h = average_edge_length(v, f).cpu().numpy() / 2
    v_ = v.detach().cpu().numpy().astype(np.double)
    f_ = f.cpu().numpy().astype(np.int32)
    v_, f_ = remesh_botsch(v_, f_, 5, h, True)
    v_ = torch.from_numpy(v_).float().cuda().contiguous()
    f_ = torch.from_numpy(f_).cuda().contiguous()
    v_, f_, _ = remove_duplicates(v_, f_)
    m = Mesh(v_, f_)
    m = auto_normals(m)
    return m

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