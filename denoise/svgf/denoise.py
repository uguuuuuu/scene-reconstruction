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


def _denoise(img, size, sigma, sigma_z, sigma_n):
    assert(img.ndim == 4)
    assert(img.shape[-1] == 7) # 3 color channels, 1 depth channel, and 3 normal channels
    res = (img.shape[2], img.shape[1])

    img = torch.permute(img, (0, 3, 1, 2)) # NHWC -> NCHW
    depth = img[:, 3:4, :, :]
    depth_grad = _img_grad(depth)
    
    result = torch.zeros_like(img[:, :3, :, :])
    weight_total = torch.zeros_like(result)

    pad = ReplicationPad2d(size)
    img = pad(img)

    p = img[:, :, size:size+res[1], size:size+res[0]]
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            q = img[:, :, size+j:size+res[1]+j, size+i:size+res[0]+i]

            w = torch.exp(torch.tensor(-(i*i + j*j) / 2 / sigma / sigma))

            w_z = torch.exp(-torch.abs(p[:,3:4,:,:] - q[:,3:4,:,:]) / 
                        (torch.abs(depth_grad[0]*i + depth_grad[1]*j) + EPSILON) / sigma_z)

            w_n = torch.pow(torch.max(torch.tensor(0.), torch.sum(p[:,4:,:,:]*q[:,4:,:,:], dim=1, keepdim=True)), sigma_n)
            
            result += q[:, :3, :, :] * w * w_z * w_n
            weight_total += w * w_z * w_n

    result = result / weight_total

    return torch.permute(result, (0, 2, 3, 1)) # NCHW -> NHWC