import torch
from torch.nn import functional, ReplicationPad2d

EPSILON = 1e-3

def _img_grad(img):
    assert(img.ndim == 4)
    device = img.device
    shape = img.shape
    res = (shape[3], shape[2])

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

    return didu, didv


def _denoise(img, depth, normal, size, sigma, sigma_z, sigma_n):
    '''
    The spatial component of Spatiotemporal Variance-Guided Filtering slightly modified
    https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced
    https://arxiv.org/abs/2206.03380

    Parameters
    ----------
    img: Tensor
        A PyTorch tensor of 4 dimensions (N, H, W, C) and 3 RGB channels
    depth: Tensor
        A PyTorch tensor of 4 dimensions (N, H, W, C) and 1 depth channel
    normal: Tensor
        A PyTorch tensor of 4 dimensions (N, H, W, C) and 3 surface normal channels    
    size: int
        Size of the filter's local footprint measured as the pixel distance from the center pixel to the rightmost pixel
    sigma: float
        Standard deviation of the Gaussian weight. A large sigma makes the effective footprint larger
    sigma_z: float
        Controls the strength of the depth edge-stopping function
    sigma_n: float
        Controls the strength of the normal edge-stopping function

    Returns
    ---------
    img_denoised:
        The loaded checkpoint
    '''
    assert(img.ndim == 4 and depth.ndim == 4 and normal.ndim == 4)
    assert(img.shape[-1] == 3 and depth.shape[-1] == 1 and normal.shape[-1] == 3) # 3 color channels, 1 depth channel, and 3 normal channels
    res = (img.shape[2], img.shape[1])

    img = torch.permute(img, (0, 3, 1, 2)) # NHWC -> NCHW
    depth = torch.permute(depth, (0, 3, 1, 2))
    normal = torch.permute(normal, (0, 3, 1, 2))
    depth_grad = _img_grad(depth)
    
    result = torch.zeros_like(img)
    weight_total = torch.zeros_like(result)

    pad = ReplicationPad2d(size)
    img = pad(img)
    depth = pad(depth)
    normal = pad(normal)

    p_d = depth[:, :, size:size+res[1], size:size+res[0]]
    p_n = normal[:, :, size:size+res[1], size:size+res[0]]
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            q = img[:, :, size+j:size+res[1]+j, size+i:size+res[0]+i]
            q_d = depth[:, :, size+j:size+res[1]+j, size+i:size+res[0]+i]
            q_n = normal[:, :, size+j:size+res[1]+j, size+i:size+res[0]+i]

            w = torch.exp(torch.tensor(-(i*i + j*j) / 2 / sigma / sigma))

            w_z = torch.exp(-torch.abs(p_d - q_d) / 
                        (torch.abs(depth_grad[0]*i + depth_grad[1]*j) + EPSILON) / sigma_z)

            w_n = torch.pow(torch.max(torch.tensor(0.), torch.sum(p_n*q_n, dim=1, keepdim=True)), sigma_n)
            
            result += q * w * w_z * w_n
            weight_total += w * w_z * w_n

    result = result / weight_total

    return torch.permute(result, (0, 2, 3, 1)) # NCHW -> NHWC