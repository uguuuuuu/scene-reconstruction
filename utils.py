from glob import glob
import pathlib
import time
import xml.etree.ElementTree as ET
import cv2
import imageio.v2 as iio
import torch
import numpy as np
import psdr_cuda
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD
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
        img = img.numpy()
    elif type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()

    img = img.reshape((res[1], res[0], img.shape[-1]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pathlib.Path(fname).suffix != '.exr':
        img = linear2srgb(img)
    cv2.imwrite(fname, img)

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

def preprocess_scene(fname, remesh_path=None, sdf_path=None):
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

    '''
    Construct source scenes
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'target')
    p[0].remove(p[1])
    p, source_shape = find_id(root, 'source')
    source_scene = ET.tostring(root, encoding='unicode')

    source_filename = replace_filename(source_shape, '')
    remesh_scene = None
    if remesh_path is not None:
        replace_filename(source_shape, remesh_path)
        remesh_scene = ET.tostring(root, encoding='unicode')
    sdf_scene = None
    if sdf_path is not None:
        replace_filename(source_shape, sdf_path)
        sdf_scene = ET.tostring(root, encoding='unicode')
    replace_filename(source_shape, source_filename)

    remove_elem(root, 'emitter')
    source_mask = ET.tostring(root, encoding='unicode')

    remesh_mask = None
    if remesh_path is not None:
        replace_filename(source_shape, remesh_path)
        remesh_mask = ET.tostring(root, encoding='unicode')
    sdf_mask = None
    if sdf_path is not None:
        replace_filename(source_shape, sdf_path)
        sdf_mask = ET.tostring(root, encoding='unicode')

    '''
    Construct target scenes
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'source')
    p[0].remove(p[1])
    target_scene = ET.tostring(root, encoding='unicode')
    remove_elem(root, 'emitter')
    target_mask = ET.tostring(root, encoding='unicode')

    return {
        'src': source_scene,
        'tgt': target_scene,
        'rm': remesh_scene,
        'sdf': sdf_scene,
        'src_mask': source_mask,
        'tgt_mask': target_mask,
        'rm_mask': remesh_mask,
        'sdf_mask': sdf_mask
    }
            
def transform(v: torch.Tensor, mat: torch.Tensor):
    v = torch.cat([v, torch.ones([v.shape[0], 1], device='cuda')], dim=-1)
    v = v @ mat.t()
    v = v[:,:3] / v[:,3:]
    return v
def transform_ek(v, mat):
    v = Vector4fD(v.x, v.y, v.z, 1.)
    v = mat @ v
    v = Vector3fD(v.x, v.y, v.z) / FloatD(v.w)
    return v
def transform_np(v, mat):
    v = np.concatenate([v, np.ones((v.shape[0], 1))], axis=-1)
    v = v @ mat.transpose()
    v = v[:,:3] / v[:,3:]
    return v

def renderC_img(xml, integrator, sensor_ids, res = (256, 256), spp = 32, load_string=True):
    scene = psdr_cuda.Scene()
    scene.load_string(xml, False) if load_string else scene.load_file(xml, False)
    scene.opts.width = res[0]
    scene.opts.height = res[1]
    scene.opts.spp = spp
    scene.opts.sppe = scene.opts.sppse = 0
    scene.opts.log_level = 0
    scene.configure()
    imgs = []
    for i in sensor_ids:
        img = integrator.renderC(scene, sensor_id=i)
        imgs.append(img.numpy())
        ek.cuda_malloc_trim()
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