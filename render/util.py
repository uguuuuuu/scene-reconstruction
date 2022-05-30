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

# def gen_tex_barycentric(samples: torch.Tensor, uvs: torch.Tensor, res: int):
#     uvs *= res
#     aabb = [torch.floor(torch.min(uvs, dim=1)[0]), torch.ceil(torch.max(uvs, dim=1)[0])]


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

'''
Adapted from Pytorch3D
https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/renderer/mesh/rasterize_meshes.html
'''
kEpsilon = 1e-8
def non_square_ndc_range(S1, S2):
    """
    In the case of non square images, we scale the NDC range
    to maintain the aspect ratio. The smaller dimension has NDC
    range of 2.0.

    Args:
        S1: dimension along with the NDC range is needed
        S2: the other image dimension

    Returns:
        ndc_range: NDC range for dimension S1
    """
    ndc_range = 2.0
    if S1 > S2:
        ndc_range = (S1 / S2) * ndc_range
    return ndc_range
def pix_to_non_square_ndc(i, S1, S2):
    """
    The default value of the NDC range is [-1, 1].
    However in the case of non square images, we scale the NDC range
    to maintain the aspect ratio. The smaller dimension has NDC
    range from [-1, 1] and the other dimension is scaled by
    the ratio of H:W.
    e.g. for image size (H, W) = (64, 128)
       Height NDC range: [-1, 1]
       Width NDC range: [-2, 2]

    Args:
        i: pixel position on axes S1
        S1: dimension along with i is given
        S2: the other image dimension

    Returns:
        pixel: NDC coordinate of point i for dimension S1
    """
    # NDC: x-offset + (i * pixel_width + half_pixel_width)
    ndc_range = non_square_ndc_range(S1, S2)
    offset = ndc_range / 2.0
    return -offset + (ndc_range * i + offset) / S1
def edge_function(p, v0, v1):
    r"""
    Determines whether a point p is on the right side of a 2D line segment
    given by the end points v0, v1.

    Args:
        p: (x, y) Coordinates of a point.
        v0, v1: (x, y) Coordinates of the end points of the edge.

    Returns:
        area: The signed area of the parallelogram given by the vectors

              .. code-block:: python

                  B = p - v0
                  A = v1 - v0

                        v1 ________
                          /\      /
                      A  /  \    /
                        /    \  /
                    v0 /______\/
                          B    p

             The area can also be interpreted as the cross product A x B.
             If the sign of the area is positive, the point p is on the
             right side of the edge. Negative area indicates the point is on
             the left side of the edge. i.e. for an edge v1 - v0

             .. code-block:: python

                             v1
                            /
                           /
                    -     /    +
                         /
                        /
                      v0
    """
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])
def barycentric_coordinates(p, v0, v1, v2):
    """
    Compute the barycentric coordinates of a point relative to a triangle.

    Args:
        p: Coordinates of a point.
        v0, v1, v2: Coordinates of the triangle vertices.

    Returns
        bary: (w0, w1, w2) barycentric coordinates in the range [0, 1].
    """
    area = edge_function(v2, v0, v1) + kEpsilon  # 2 x face area.
    w0 = edge_function(p, v1, v2) / area
    w1 = edge_function(p, v2, v0) / area
    w2 = edge_function(p, v0, v1) / area
    return (w0, w1, w2)
def rasterize(verts: torch.Tensor, faces: torch.Tensor, res=(256, 256), faces_per_pixel=8, perspective_correct=False):
    '''
    Rasterize the given triangles. X points to the left, Y points to the up, and Z points inward the screen

    Args:
        verts: Vertex positions in the aforementioned coordinate system.
        faces: `|F| x 3` tensor of vertex indices.
        res: Resolution of the resulting image.
        faces_per_pixel: Indices of the closest `K` faces stored per pixel in terms of their depths where `K == faces_per_pixel`
        perspective_correct: Whether to perspective-correct the barycentric coordinate of each pixel

    Returns
        face_idxs: `(H, W, K)` tensor storing the faces each pixel belongs to where `K == faces_per_pixel`
        bary_coords: `(H, W, K, 3)` tensor storing the barycentric coordinates of each pixel in each face they belong to
    '''

    H, W = res
    K = faces_per_pixel

    device = verts.device

    faces_verts = verts[faces.long()]
    
    # Initialize output tensors.
    face_idxs = torch.full(
        (H, W, K), fill_value=-1, dtype=torch.int64, device=device
    )
    bary_coords = torch.full(
        (H, W, K, 3), fill_value=-1, dtype=torch.float32, device=device
    )

    # Calculate all face bounding boxes.
    x_mins = torch.min(faces_verts[:, :, 0], dim=1, keepdim=True).values
    x_maxs = torch.max(faces_verts[:, :, 0], dim=1, keepdim=True).values
    y_mins = torch.min(faces_verts[:, :, 1], dim=1, keepdim=True).values
    y_maxs = torch.max(faces_verts[:, :, 1], dim=1, keepdim=True).values
    z_mins = torch.min(faces_verts[:, :, 2], dim=1, keepdim=True).values
    print(x_mins.shape)

    # Iterate through the horizontal lines of the image from top to bottom.
    for yi in range(H):
        # Y coordinate of one end of the image. Reverse the ordering
        # of yi so that +Y is pointing up in the image.
        yfix = H - 1 - yi
        yf = pix_to_non_square_ndc(yfix, H, W)
        # Iterate through pixels on this horizontal line, left to right.
        for xi in range(W):
            # X coordinate of one end of the image. Reverse the ordering
            # of xi so that +X is pointing to the left in the image.
            xfix = W - 1 - xi
            xf = pix_to_non_square_ndc(xfix, W, H)
            top_k_points = []
            # Check whether each face in the mesh affects this pixel.
            for f in range(faces.shape[0]):
                face = faces_verts[f]
                v0, v1, v2 = face.unbind(0)
                face_area = edge_function(v0, v1, v2)
                # Ignore faces which have zero area.
                if face_area == 0.0:
                    continue
                outside_bbox = (
                    xf < x_mins[f]
                    or xf > x_maxs[f]
                    or yf < y_mins[f]
                    or yf > y_maxs[f]
                )
                # Faces with at least one vertex behind the camera won't
                # render correctly and should be removed or clipped before
                # calling the rasterizer
                if z_mins[f] < kEpsilon:
                    continue
                # Check if pixel is outside of face bbox.
                if outside_bbox:
                    continue
                # Compute barycentric coordinates and pixel z distance.
                pxy = torch.tensor([xf, yf], dtype=torch.float32, device=device)
                bary = barycentric_coordinates(pxy, v0[:2], v1[:2], v2[:2])
                if perspective_correct:
                        z0, z1, z2 = v0[2], v1[2], v2[2]
                        l0, l1, l2 = bary[0], bary[1], bary[2]
                        top0 = l0 * z1 * z2
                        top1 = z0 * l1 * z2
                        top2 = z0 * z1 * l2
                        bot = top0 + top1 + top2
                        bary = torch.stack([top0 / bot, top1 / bot, top2 / bot])

                inside = all(x > 0.0 for x in bary)
                if not inside:
                    continue

                pz = bary[0] * v0[2] + bary[1] * v1[2] + bary[2] * v2[2]
                # Check if point is behind the image.
                if pz < 0:
                    continue

                # Handle the case where a face (f) partially behind the image plane is
                # clipped to a quadrilateral and then split into two faces (t1, t2).
                top_k_points.append((pz, f, bary))
                top_k_points.sort()
                if len(top_k_points) > K:
                    top_k_points = top_k_points[:K]
            # Save to output tensors.
            for k, (pz, f, bary) in enumerate(top_k_points):
                face_idxs[yi, xi, k] = f
                bary_coords[yi, xi, k, 0] = bary[0]
                bary_coords[yi, xi, k, 1] = bary[1]
                bary_coords[yi, xi, k, 2] = bary[2]

    return face_idxs, bary_coords