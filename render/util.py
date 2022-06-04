from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD
import torch
from torch.nn import functional
import cv2
import numpy as np
import pathlib

def transform(v, mat):
    v = Vector4fD(v.x, v.y, v.z, 1.)
    v = mat @ v
    v = Vector3fD(v.x, v.y, v.z) / FloatD(v.w)
    return v

'''
Adapted from https://github.com/NVlabs/nvdiffrec
'''
def sample(img: torch.Tensor, uvs: torch.Tensor, filter='nearest'):
    img = img[None, ...] # Add batch dimension
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
    samples = functional.grid_sample(img, uvs[None, None, ...]*2-1,
                mode=filter, align_corners=False)
    samples = samples.permute(0, 2, 3, 1) # NCHW -> NHWC
    return samples[0, 0, ...]
def scale_img(x, size, mag='bilinear', min='area'):
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    if len(x.shape) == 3:
        return scale_img_nhwc(x[None,...], size, mag, min)[0]
    else:
        return scale_img_nhwc(x, size, mag, min)
def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def wrap(uvs):
    if type(uvs) == np.ndarray:
        return wrap_np(uvs)
    offset = uvs - uvs.floor()
    uvs_ = torch.where(uvs < 0, offset, uvs)
    uvs_ = torch.where(uvs_ > 1, offset, uvs_)
    return uvs_
def flip_x(uv):
    if type(uv) == np.ndarray:
        return flip_x_np(uv)
    flipped = -uv[...,0]
    flipped[flipped==0] = 1
    return torch.stack([flipped, uv[...,1]], dim=-1)
def flip_y(uv):
    if type(uv) == np.ndarray:
        return flip_y_np(uv)
    flipped = -uv[...,1]
    flipped[flipped==0] = 1
    return torch.stack([uv[...,0], flipped], dim=-1)
def wrap_np(uvs: np.ndarray):
    offset = uvs - np.floor(uvs)
    uvs_ = np.where(uvs < 0, offset, uvs)
    uvs_ = np.where(uvs_ > 1, offset, uvs_)
    return uvs_
def flip_x_np(uv):
    flipped = -uv[...,0]
    flipped[flipped==0] = 1
    return np.stack([flipped, uv[...,1]], axis=-1)
def flip_y_np(uv):
    flipped = -uv[...,1]
    flipped[flipped==0] = 1
    return np.stack([uv[...,0], flipped], axis=-1)
    
def gen_tex(samples: torch.Tensor, uvs: torch.Tensor, res: int):
    assert(samples.shape[0] == uvs.shape[0])
    n_channels = samples.shape[-1]
    uvs = wrap(uvs)
    uvs *= res
    texels = torch.floor(uvs).to(torch.long)
    texels = texels.clamp(0, res-1)
    texels = texels[:,0] + texels[:,1]*res
    tex = torch.zeros([res*res, n_channels], device='cuda')
    tex = tex.scatter_add(0, texels.reshape(-1,1).expand(-1,n_channels), samples)
    total_weights = torch.zeros(tex.shape[0], device='cuda').scatter_add(0, texels,
                    torch.ones_like(texels, device='cuda', dtype=torch.float))
    tex = tex / total_weights.reshape(-1,1)
    return tex.reshape(res, res, n_channels)

def linear2srgb(img):
    img[img <= 0.0031308] *= 12.92
    img[img > 0.0031308] = 1.055*img[img > 0.0031308]**(1./2.4) - 0.055
    img = np.uint16(np.clip(img, 0., 1.)*65535)
    return img
def save_img(img, fname):
    img = img.detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pathlib.Path(fname).suffix != '.exr':
        img = linear2srgb(img)
    cv2.imwrite(fname, img)

