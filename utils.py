from glob import glob
import pathlib
import time
import cv2
import torch
from torch.nn import functional
import numpy as np
import psdr_cuda
import enoki as ek
from enoki.cuda_autodiff import Vector3f as Vector3fD, Float32 as FloatD
from enoki.cuda import Vector3f as Vector3fC

def linear2srgb(img):
    shape = img.shape
    img = img.flatten()
    img[img <= 0.0031308] *= 12.92
    img[img > 0.0031308] = 1.055*img[img > 0.0031308]**(1./2.4) - 0.055
    img = np.uint16(np.clip(img, 0., 1.)*65535)
    return img.reshape(shape)

def save_img(img, fname, scene):
    i = 0.
    if type(img) == Vector3fD or type(img) == Vector3fC:
        i = img.numpy().reshape((scene.opts.height, scene.opts.width, 3))
    else:
        i = img.detach().cpu().numpy().reshape((scene.opts.height, scene.opts.width, 3))
    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
    if pathlib.Path(fname).suffix != '.exr':
        i = linear2srgb(i)
    cv2.imwrite(fname, i)

def exr2png(fname):
    img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = linear2srgb(img)
    cv2.imwrite(str(pathlib.Path(fname).with_suffix('.png')), img)

class NotFoundError(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)
def load_tensor(pattern, n_itr = None):
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

    if n_itr is None:
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
            

def renderC_img(fname, integrator, sensor_ids, res = (256, 256), spp = 32):
    scene = psdr_cuda.Scene()
    scene.load_file(fname, False)
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

def get_loss_fn(loss_fn):
    if loss_fn == 'l1':
        return functional.l1_loss
    elif loss_fn == 'l2':
        return functional.mse_loss
    else:
        raise NotImplementedError(f'Loss function {loss_fn} not implemented yet')

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
    faces : enoki.cuda_autodiff.Vector3i
        array of triangle faces.
    """

    faces = faces.torch()

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

'''
    Adapted from the "Large Steps in Inverse Rendering of Geometry" paper
    https://rgl.epfl.ch/publications/Nicolet2021Large
'''
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
    if sqr:
        # return 0.5 * l * torch.trace(torch.sparse.mm(v.transpose(0, 1), laplacian)
        #                             @ torch.sparse.mm(laplacian, v))
        return (L@v).square().mean()
    else:
        # return 0.5 * l * torch.trace(v.transpose(0, 1) @ torch.sparse.mm(laplacian, v))
        return (v * (L@v)).mean()

class RenderD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, v):
        assert(v.requires_grad)
        scene, key, integrator, sensor_ids = \
                    [params[k] for k in ['scene', 'key', 'integrator', 'sensor_ids']]

        ctx.v = Vector3fD(v)
        ek.set_requires_gradient(ctx.v)
        scene.param_map[key].vertex_positions = ctx.v
        scene.configure()
        ctx.out = []
        for id in sensor_ids:
            ctx.out.append(integrator.renderD(scene, id))

        out = torch.stack([img.torch() for img in ctx.out])
        # ek.cuda_malloc_trim()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        for i in range(len(ctx.out)):
            ek.set_gradient(ctx.out[i], Vector3fC(grad_out[i]))
        FloatD.backward()
        grad = ek.gradient(ctx.v).torch()

        del ctx.v, ctx.out
        # ek.cuda_malloc_trim()
        return (None, grad)
renderD = RenderD.apply

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