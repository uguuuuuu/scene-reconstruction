import psdr_cuda
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector2f as Vector2fD, Vector3f as Vector3fD, Vector3i as Vector3iD
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
        ctx.masks = [ek.hmean(integrator_mask.renderD(scene, id)) for id in sensor_ids]

        imgs = torch.stack([img.torch() for img in ctx.imgs])
        masks = torch.stack([mask.torch() for mask in ctx.masks])
        masks = masks[...,None]

        # ek.cuda_malloc_trim()
        return torch.cat([imgs, masks], dim=-1)

    @staticmethod
    def backward(ctx, grad_out):
        for i in range(len(ctx.imgs)):
            ek.set_gradient(ctx.imgs[i], Vector3fC(grad_out[i,:,0:3]))
            ek.set_gradient(ctx.masks[i], FloatC(grad_out[i,:,3]))
        FloatD.backward()

        grad_v = ek.gradient(ctx.v)
        nan_mask = ek.isnan(grad_v)
        grad_v = ek.select(nan_mask, 0, grad_v).torch()
        grad_mat = ek.gradient(ctx.mat)
        nan_mask = ek.isnan(grad_mat)
        grad_mat = ek.select(nan_mask, 0, grad_mat).torch()

        del ctx.v, ctx.mat, ctx.imgs, ctx.masks
        # ek.cuda_malloc_trim()
        return grad_v, grad_mat
renderDVAM = RenderD_Vertex_Alpha_Mat.apply

class RenderD_Vertex_Alpha_Mat_Env(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, mat, env):
        assert(v.requires_grad and mat.requires_grad and env.requires_grad)
        scene = config.scene
        integrator, integrator_mask =  config.integrator, config.integrator_mask
        key, sensor_ids = config.key, config.sensor_ids

        bsdf_type = type(scene.param_map[key].bsdf)

        ctx.v = Vector3fD(v)
        ctx.env = Vector3fD(env)
        ek.set_requires_gradient(ctx.v)
        ek.set_requires_gradient(ctx.env)
        scene.param_map[key].vertex_positions = ctx.v
        scene.param_map['Emitter[0]'].radiance.data = ctx.env
        if bsdf_type == psdr_cuda.DiffuseBSDF:
            ctx.albedo = Vector3fD(mat.reshape(-1,3))
            ek.set_requires_gradient(ctx.albedo)
            scene.param_map[key].bsdf.reflectance.data = ctx.albedo
        else:
            ctx.alpha_u = FloatD(mat[...,0:1].reshape(-1))
            ctx.alpha_v = FloatD(mat[...,1:2].reshape(-1))
            ctx.eta = Vector3fD(mat[...,2:5].reshape(-1,3))
            ctx.k = Vector3fD(mat[...,5:8].reshape(-1,3))
            ctx.specular_reflectance = Vector3fD(mat[...,8:].reshape(-1,3))
            ek.set_requires_gradient(ctx.alpha_u)
            ek.set_requires_gradient(ctx.alpha_v)
            ek.set_requires_gradient(ctx.eta)
            ek.set_requires_gradient(ctx.k)
            ek.set_requires_gradient(ctx.specular_reflectance)
            scene.param_map[key].bsdf.alpha_u.data = ctx.alpha_u
            scene.param_map[key].bsdf.alpha_v.data = ctx.alpha_v
            scene.param_map[key].bsdf.eta.data = ctx.eta
            scene.param_map[key].bsdf.k.data = ctx.k
            scene.param_map[key].bsdf.specular_reflectance.data = ctx.specular_reflectance
        scene.configure()
        
        ctx.imgs = [integrator.renderD(scene, id) for id in sensor_ids]
        ctx.masks = [ek.hmean(integrator_mask.renderD(scene, id)) for id in sensor_ids]

        imgs = torch.stack([img.torch() for img in ctx.imgs])
        masks = torch.stack([mask.torch() for mask in ctx.masks])
        masks = masks[...,None]
        
        # ek.cuda_malloc_trim()
        return torch.cat([imgs, masks], dim=-1)

    @staticmethod
    def backward(ctx, grad_out):
        for i in range(len(ctx.imgs)):
            ek.set_gradient(ctx.imgs[i], Vector3fC(grad_out[i,:,0:3]))
            ek.set_gradient(ctx.masks[i], FloatC(grad_out[i,:,3]))
        FloatD.backward()

        bsdf_type = type(config.scene.param_map[config.key].bsdf)

        grad_v = ek.gradient(ctx.v)
        nan_mask = ek.isnan(grad_v)
        grad_v = ek.select(nan_mask, 0, grad_v).torch()

        grad_env = ek.gradient(ctx.env)
        nan_mask = ek.isnan(grad_env)
        grad_env = ek.select(nan_mask, 0, grad_env).torch()

        if bsdf_type == psdr_cuda.DiffuseBSDF:
            grad_albedo = ek.gradient(ctx.albedo)
            nan_mask = ek.isnan(grad_albedo)
            grad_albedo = ek.select(nan_mask, 0, grad_albedo).torch()
            grad_mat = grad_albedo
        else:
            grad_alpha_u = ek.gradient(ctx.alpha_u)
            grad_alpha_v = ek.gradient(ctx.alpha_v)
            grad_eta = ek.gradient(ctx.eta)
            grad_k = ek.gradient(ctx.k)
            grad_specular_reflectance = ek.gradient(ctx.specular_reflectance)
            nan_mask = ek.isnan(grad_alpha_u)
            grad_alpha_u = ek.select(nan_mask, 0, grad_alpha_u).torch()
            nan_mask = ek.isnan(grad_alpha_v)
            grad_alpha_v = ek.select(nan_mask, 0, grad_alpha_v).torch()
            nan_mask = ek.isnan(grad_eta)
            grad_eta = ek.select(nan_mask, 0, grad_eta).torch()
            nan_mask = ek.isnan(grad_k)
            grad_k = ek.select(nan_mask, 0, grad_k).torch()
            nan_mask = ek.isnan(grad_specular_reflectance)
            grad_specular_reflectance = ek.select(nan_mask, 0, grad_specular_reflectance).torch()
            grad_mat = torch.cat([grad_alpha_u[...,None], grad_alpha_v[...,None], grad_eta, grad_k, grad_specular_reflectance], dim=-1)

        # ek.cuda_malloc_trim()
        result = grad_v, grad_mat, grad_env

        max, min = torch.max(result[1]), torch.min(result[1])
        if max > 1000 or min < -1000:
            print(max, min)
        # msg = ''
        # if torch.isinf(max):
        #     msg += 'inf' + ' '
        # if torch.isinf(min):
        #     msg += '-inf'
        # if msg != '': print(msg) 
        
        result[1].clamp_(-10000, 10000)

        del ctx.v, ctx.env, ctx.imgs, ctx.masks
        if bsdf_type == psdr_cuda.DiffuseBSDF:
            del ctx.albedo
        else:
            del ctx.alpha_u, ctx.alpha_v, ctx.eta, ctx.k, ctx.specular_reflectance

        return result
renderDVAME = RenderD_Vertex_Alpha_Mat_Env.apply