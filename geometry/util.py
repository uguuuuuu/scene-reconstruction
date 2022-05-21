######################################################################################
# Adapted from https://github.com/NVlabs/nvdiffrec
######################################################################################

import os
import numpy as np
import torch
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD
from pyremesh import remesh_botsch

#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

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
    return v_, f_