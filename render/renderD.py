import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector3i as Vector3iD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC
import torch

from . import config

class RenderD_Vert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        assert(v.requires_grad)
        scene, key, integrator, sensor_ids = \
            config.scene, config.key, config.integrator, config.sensor_ids

        ctx.v = Vector3fD(v)
        ek.set_requires_gradient(ctx.v)
        scene.param_map[key].vertex_positions = ctx.v
        scene.configure()
        ctx.out = [integrator.renderD(scene, id) for id in sensor_ids]

        out = torch.stack([img.torch() for img in ctx.out])
        # ek.cuda_malloc_trim()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        for i in range(len(ctx.out)):
            ek.set_gradient(ctx.out[i], Vector3fC(grad_out[i]))
        FloatD.backward()

        grad = ek.gradient(ctx.v)
        nan_mask = ek.isnan(grad)
        grad = ek.select(nan_mask, 0, grad).torch()

        del ctx.v, ctx.out
        # ek.cuda_malloc_trim()
        return grad
renderDV = RenderD_Vert.apply

class RenderD_Vert_Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        assert(v.requires_grad)
        scene = config.scene
        integrator, integrator_mask =  config.integrator, config.integrator_mask
        key, sensor_ids = config.key, config.sensor_ids

        ctx.v = Vector3fD(v)
        ek.set_requires_gradient(ctx.v)
        scene.param_map[key].vertex_positions = ctx.v
        scene.configure()
        ctx.imgs = [integrator.renderD(scene, id) for id in sensor_ids]
        ctx.masks = [integrator_mask.renderD(scene, id) for id in sensor_ids]

        imgs = torch.stack([img.torch() for img in ctx.imgs])
        masks = torch.stack([mask.torch() for mask in ctx.masks])
        # ek.cuda_malloc_trim()
        return torch.stack([imgs, masks])

    @staticmethod
    def backward(ctx, grad_out):
        for i in range(len(ctx.imgs)):
            ek.set_gradient(ctx.imgs[i], Vector3fC(grad_out[0][i]))
            ek.set_gradient(ctx.masks[i], Vector3fC(grad_out[1][i]))
        FloatD.backward()

        grad = ek.gradient(ctx.v)
        nan_mask = ek.isnan(grad)
        grad = ek.select(nan_mask, 0, grad).torch()

        del ctx.v, ctx.imgs, ctx.masks
        # ek.cuda_malloc_trim()
        return grad
renderDVA = RenderD_Vert_Alpha.apply

class RenderD_Mat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat):
        assert(mat.requires_grad)
        scene, key, integrator, sensor_ids = \
            config.scene, config.key, config.integrator, config.sensor_ids

        ctx.mat = Vector3fD(mat)
        ek.set_requires_gradient(ctx.mat)
        scene.param_map[key].bsdf.reflectance.data = ctx.mat
        scene.configure()
        ctx.out = [integrator.renderD(scene, id) for id in sensor_ids]

        out = torch.stack([img.torch() for img in ctx.out])
        # ek.cuda_malloc_trim()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        for i in range(len(ctx.out)):
            ek.set_gradient(ctx.out[i], Vector3fC(grad_out[i]))
        FloatD.backward()

        grad = ek.gradient(ctx.mat)
        nan_mask = ek.isnan(grad)
        grad = ek.select(nan_mask, 0, grad).torch()

        del ctx.mat, ctx.out
        # ek.cuda_malloc_trim()
        return grad
renderDM = RenderD_Mat.apply

class RenderD_Vertex_Alpha_Mat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, mat):
        assert(v.requires_grad and mat.requires_grad)
        scene = config.scene
        integrator, integrator_mask =  config.integrator, config.integrator_mask
        key, sensor_ids = config.key, config.sensor_ids

        ctx.v = Vector3fD(v)
        ctx.mat = Vector3fD(mat)
        ek.set_requires_gradient(ctx.v)
        ek.set_requires_gradient(ctx.mat)
        scene.param_map[key].vertex_positions = ctx.v
        scene.param_map[key].bsdf.reflectance.data = ctx.mat
        scene.configure()
        ctx.imgs = [integrator.renderD(scene, id) for id in sensor_ids]
        ctx.masks = [integrator_mask.renderD(scene, id) for id in sensor_ids]

        imgs = torch.stack([img.torch() for img in ctx.imgs])
        masks = torch.stack([mask.torch() for mask in ctx.masks])
        # ek.cuda_malloc_trim()
        return torch.stack([imgs, masks])

    @staticmethod
    def backward(ctx, grad_out):
        for i in range(len(ctx.imgs)):
            ek.set_gradient(ctx.imgs[i], Vector3fC(grad_out[0][i]))
            ek.set_gradient(ctx.masks[i], Vector3fC(grad_out[1][i]))
        FloatD.backward()

        grad_v = ek.gradient(ctx.v)
        nan_mask = ek.isnan(grad_v)
        grad_v = ek.select(nan_mask, 0, grad_v).torch()
        grad_mat = ek.gradient(ctx.mat)
        nan_mask = ek.isnan(grad_mat)
        grad_mat = ek.select(nan_mask, 0, grad_mat).torch()

        del ctx.v, ctx.imgs, ctx.masks
        # ek.cuda_malloc_trim()
        return grad_v, grad_mat