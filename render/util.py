from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD
import torch
from torch.nn import functional

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

def wrap(uvs: torch.Tensor):
    dirty = False
    if (uvs > 1).any():
        uvs[uvs>1] -= 1
        dirty = True
    if (uvs < 0).any():
        uvs[uvs<0] += 1
        dirty = True
    if dirty:
        wrap(uvs)
    
def gen_tex(samples: torch.Tensor, uvs: torch.Tensor, res: int):
    assert(samples.shape[0] == uvs.shape[0])
    n_channels = samples.shape[-1]
    wrap(uvs)
    uvs *= res
    texels = torch.floor(uvs).to(torch.long)
    texels = texels.clamp(0, res-1)
    texels = texels[:,0] + texels[:,1]*res
    tex = torch.zeros([res*res, n_channels], device='cuda')
    tex = tex.scatter_add(0, texels.reshape(-1,1).expand(-1,n_channels), samples)
    total_weights = torch.zeros(tex.shape[0], device='cuda').scatter_add(0, texels,
                    torch.ones_like(texels, device='cuda', dtype=torch.float))
    return tex / total_weights.reshape(-1,1)