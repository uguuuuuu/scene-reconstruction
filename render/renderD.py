import psdr_cuda
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector2f as Vector2fD, Vector3f as Vector3fD, Vector3i as Vector3iD
from enoki.cuda import Float32 as FloatC, Vector3f as Vector3fC
import torch

from . import config

EPSILON = 1e-5

class RenderD(torch.autograd.Function):
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
            ctx.albedo = Vector3fD(mat)
            ek.set_requires_gradient(ctx.albedo)
            scene.param_map[key].bsdf.reflectance.data = ctx.albedo
        else:
            ctx.alpha = FloatD(mat[...,0])
            ctx.eta = Vector3fD(mat[...,1:4])
            ctx.k = Vector3fD(mat[...,4:7])
            ctx.specular_reflectance = Vector3fD(mat[...,7:])
            ek.set_requires_gradient(ctx.alpha)
            ek.set_requires_gradient(ctx.eta)
            ek.set_requires_gradient(ctx.k)
            ek.set_requires_gradient(ctx.specular_reflectance)
            scene.param_map[key].bsdf.alpha_u.data = ctx.alpha
            scene.param_map[key].bsdf.alpha_v.data = ctx.alpha
            scene.param_map[key].bsdf.eta.data = ctx.eta
            scene.param_map[key].bsdf.k.data = ctx.k
            scene.param_map[key].bsdf.specular_reflectance.data = ctx.specular_reflectance
        scene.configure()
        
        ctx.imgs = [integrator.renderD(scene, id) for id in sensor_ids]
        ctx.masks = [integrator_mask.renderD(scene, id) for id in sensor_ids]

        imgs = torch.stack([img.torch() for img in ctx.imgs])
        masks = torch.stack([mask.torch() for mask in ctx.masks])
        
        # ek.cuda_malloc_trim()
        return imgs, masks

    @staticmethod
    def backward(ctx, *grad_out):
        for i in range(len(ctx.imgs)):
            ek.set_gradient(ctx.imgs[i], Vector3fC(grad_out[0][i]))
            ek.set_gradient(ctx.masks[i], Vector3fC(grad_out[1][i]))
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
            grad_alpha = ek.gradient(ctx.alpha)
            grad_eta = ek.gradient(ctx.eta)
            grad_k = ek.gradient(ctx.k)
            grad_specular_reflectance = ek.gradient(ctx.specular_reflectance)

            nan_mask = ek.isnan(grad_alpha)
            grad_alpha = ek.select(nan_mask, 0, grad_alpha).torch()
            grad_alpha = torch.where(torch.isfinite(grad_alpha), grad_alpha, torch.zeros_like(grad_alpha))
            grad_alpha = grad_alpha / 100.
            nan_mask = ek.isnan(grad_eta)
            grad_eta = ek.select(nan_mask, 0, grad_eta).torch()
            nan_mask = ek.isnan(grad_k)
            grad_k = ek.select(nan_mask, 0, grad_k).torch()
            nan_mask = ek.isnan(grad_specular_reflectance)
            grad_specular_reflectance = ek.select(nan_mask, 0, grad_specular_reflectance).torch()

            max, min = torch.max(grad_alpha), torch.min(grad_alpha)
            if max > 1000 or min < -1000:
                print('grad_alpha', max, min)
            max, min = torch.max(grad_eta), torch.min(grad_eta)
            if max > 1000 or min < -1000:
                print('grad_eta', max, min)
            max, min = torch.max(grad_k), torch.min(grad_k)
            if max > 1000 or min < -1000:
                print('grad_k', max, min)
            max, min = torch.max(grad_specular_reflectance), torch.min(grad_specular_reflectance)
            if max > 1000 or min < -1000:
                print('grad_specular_reflectance', max, min)

            grad_mat = torch.cat([grad_alpha[...,None], grad_eta, grad_k, grad_specular_reflectance], dim=-1)

        # ek.cuda_malloc_trim()
        result = grad_v, grad_mat, grad_env

        # msg = ''
        # if torch.isinf(max):
        #     msg += 'inf' + ' '
        # if torch.isinf(min):
        #     msg += '-inf'
        # if msg != '': print(msg) 
        
        result[1].clamp_(-1000, 1000)

        del ctx.v, ctx.env, ctx.imgs, ctx.masks
        if bsdf_type == psdr_cuda.DiffuseBSDF:
            del ctx.albedo
        else:
            del ctx.alpha, ctx.eta, ctx.k, ctx.specular_reflectance

        return result
renderD = RenderD.apply

class RenderD_Demodulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, mat, env):
        assert(v.requires_grad and mat.requires_grad and env.requires_grad)
        scene = config.scene
        integrator_demod, integrator_mask = config.integrator_demod, config.integrator_mask
        integrator_alb, integrator_depth = config.integrator_alb, config.integrator_depth
        integrator_normal = config.integrator_normal
        key, sensor_ids = config.key, config.sensor_ids

        bsdf_type = type(scene.param_map[key].bsdf)

        ctx.v = Vector3fD(v)
        ctx.env = Vector3fD(env)
        ek.set_requires_gradient(ctx.v)
        ek.set_requires_gradient(ctx.env)
        scene.param_map[key].vertex_positions = ctx.v
        scene.param_map['Emitter[0]'].radiance.data = ctx.env
        if bsdf_type == psdr_cuda.DiffuseBSDF:
            ctx.albedo = Vector3fD(mat)
            ek.set_requires_gradient(ctx.albedo)
            scene.param_map[key].bsdf.reflectance.data = ctx.albedo
        elif bsdf_type == psdr_cuda.RoughConductorBSDF:
            ctx.alpha = FloatD(mat[...,0])
            ctx.eta = Vector3fD(mat[...,1:4])
            ctx.k = Vector3fD(mat[...,4:7])
            ctx.specular_reflectance = Vector3fD(mat[...,7:])
            ek.set_requires_gradient(ctx.alpha)
            ek.set_requires_gradient(ctx.eta)
            ek.set_requires_gradient(ctx.k)
            ek.set_requires_gradient(ctx.specular_reflectance)
            scene.param_map[key].bsdf.alpha_u.data = ctx.alpha
            scene.param_map[key].bsdf.alpha_v.data = ctx.alpha
            scene.param_map[key].bsdf.eta.data = ctx.eta
            scene.param_map[key].bsdf.k.data = ctx.k
            scene.param_map[key].bsdf.specular_reflectance.data = ctx.specular_reflectance
        else:
            raise NotImplementedError('Unsupported BSDF: ' + bsdf_type.__name__)
        scene.configure()
        
        ctx.imgs_demod = [integrator_demod.renderD(scene, id) for id in sensor_ids]
        ctx.masks = [integrator_mask.renderD(scene, id) for id in sensor_ids]
        ctx.imgs_alb = [integrator_alb.renderD(scene, id) for id in sensor_ids]
        ctx.imgs_depth = [integrator_depth.renderD(scene, id) for id in sensor_ids]
        ctx.imgs_normal = [integrator_normal.renderD(scene, id) for id in sensor_ids]

        imgs_demod = torch.stack([img.torch() for img in ctx.imgs_demod])
        masks = torch.stack([mask.torch() for mask in ctx.masks])
        imgs_alb = torch.stack([alb.torch() for alb in ctx.imgs_alb])
        imgs_depth = torch.stack([depth.torch() for depth in ctx.imgs_depth])
        imgs_normal = torch.stack([normal.torch() for normal in ctx.imgs_normal])
        
        ek.cuda_malloc_trim()
        return imgs_demod, masks, imgs_alb, imgs_depth, imgs_normal

    @staticmethod
    def backward(ctx, *grad_out):
        for i in range(len(ctx.imgs_demod)):
            ek.set_gradient(ctx.imgs_demod[i], Vector3fC(grad_out[0][i]))
            ek.set_gradient(ctx.masks[i], Vector3fC(grad_out[1][i]))
            ek.set_gradient(ctx.imgs_alb[i], Vector3fC(grad_out[2][i]))
            ek.set_gradient(ctx.imgs_depth[i], Vector3fC(grad_out[3][i]))
            ek.set_gradient(ctx.imgs_normal[i], Vector3fC(grad_out[4][i]))
        FloatD.backward()

        bsdf_type = type(config.scene.param_map[config.key].bsdf)

        grad_v = ek.gradient(ctx.v)
        nan_mask = ek.isnan(grad_v)
        grad_v = ek.select(nan_mask, 0, grad_v).torch()
        max, min = torch.max(grad_v), torch.min(grad_v)
        if max > 1000 or min < -1000:
            print('grad_v', max, min)
        grad_v = torch.where(torch.isfinite(grad_v), grad_v, 0)

        grad_env = ek.gradient(ctx.env)
        nan_mask = ek.isnan(grad_env)
        grad_env = ek.select(nan_mask, 0, grad_env).torch()
        max, min = torch.max(grad_env), torch.min(grad_env)
        if max > 1000 or min < -1000:
            print('grad_env', max, min)
        grad_env = torch.where(torch.isfinite(grad_env), grad_env, 0)

        if bsdf_type == psdr_cuda.DiffuseBSDF:
            grad_albedo = ek.gradient(ctx.albedo)
            nan_mask = ek.isnan(grad_albedo)
            grad_albedo = ek.select(nan_mask, 0, grad_albedo).torch()
            grad_mat = grad_albedo

            max, min = torch.max(grad_albedo), torch.min(grad_albedo)
            if max > 1000 or min < -1000:
                print('grad_albedo', max, min)
        elif bsdf_type == psdr_cuda.RoughConductorBSDF:
            grad_alpha = ek.gradient(ctx.alpha)
            grad_eta = ek.gradient(ctx.eta)
            grad_k = ek.gradient(ctx.k)
            grad_specular_reflectance = ek.gradient(ctx.specular_reflectance)

            nan_mask = ek.isnan(grad_alpha)
            grad_alpha = ek.select(nan_mask, 0, grad_alpha).torch()
            grad_alpha = torch.where(torch.isfinite(grad_alpha), grad_alpha, torch.zeros_like(grad_alpha))
            grad_alpha = grad_alpha / 100.
            nan_mask = ek.isnan(grad_eta)
            grad_eta = ek.select(nan_mask, 0, grad_eta).torch()
            nan_mask = ek.isnan(grad_k)
            grad_k = ek.select(nan_mask, 0, grad_k).torch()
            nan_mask = ek.isnan(grad_specular_reflectance)
            grad_specular_reflectance = ek.select(nan_mask, 0, grad_specular_reflectance).torch()

            max, min = torch.max(grad_alpha), torch.min(grad_alpha)
            if max > 1000 or min < -1000:
                print('grad_alpha', max, min)
            max, min = torch.max(grad_eta), torch.min(grad_eta)
            if max > 1000 or min < -1000:
                print('grad_eta', max, min)
            max, min = torch.max(grad_k), torch.min(grad_k)
            if max > 1000 or min < -1000:
                print('grad_k', max, min)
            max, min = torch.max(grad_specular_reflectance), torch.min(grad_specular_reflectance)
            if max > 1000 or min < -1000:
                print('grad_specular_reflectance', max, min)

            grad_mat = torch.cat([grad_alpha[...,None], grad_eta, grad_k, grad_specular_reflectance], dim=-1)
        else:
            raise NotImplementedError('Unsupported BSDF: ' + bsdf_type.__name__)

        result = grad_v, grad_mat, grad_env

        # msg = ''
        # if torch.isinf(max):
        #     msg += 'inf' + ' '
        # if torch.isinf(min):
        #     msg += '-inf'
        # if msg != '': print(msg) 
        
        result[1].clamp_(-1000, 1000)

        del ctx.v, ctx.env, ctx.imgs_demod, ctx.masks
        if bsdf_type == psdr_cuda.DiffuseBSDF:
            del ctx.albedo
        else:
            del ctx.alpha, ctx.eta, ctx.k, ctx.specular_reflectance

        ek.cuda_malloc_trim()
        return result
renderD_demod = RenderD_Demodulate.apply

# class RenderD_Albedo(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, v, alb, env, update_param=True):
#         assert(v.requires_grad and alb.requires_grad and env.requires_grad)
#         scene = config.scene
#         integrator_uv, integrator_mask = config.integrator_uv, config.integrator_mask
#         key, sensor_ids = config.key, config.sensor_ids

#         bsdf_type = type(scene.param_map[key].bsdf)

#         if update_param:
#             ctx.v = Vector3fD(v)
#             ctx.env = Vector3fD(env)
#             ek.set_requires_gradient(ctx.v)
#             ek.set_requires_gradient(ctx.env)
#             scene.param_map[key].vertex_positions = ctx.v
#             scene.param_map['Emitter[0]'].radiance.data = ctx.env
#             if bsdf_type == psdr_cuda.DiffuseBSDF:
#                 ctx.albedo = Vector3fD(alb)
#                 ek.set_requires_gradient(ctx.albedo)
#                 scene.param_map[key].bsdf.reflectance.data = ctx.albedo
#             else:
#                 ctx.specular_reflectance = Vector3fD(alb)
#                 ek.set_requires_gradient(ctx.specular_reflectance)
#                 scene.param_map[key].bsdf.specular_reflectance.data = ctx.specular_reflectance
#         else:
#             ctx.v = scene.param_map[key].vertex_positions
#             ctx.env = scene.param_map['Emitter[0]'].radiance.data
#             if bsdf_type == psdr_cuda.DiffuseBSDF:
#                 ctx.albedo = scene.param_map[key].bsdf.reflectance.data
#             else:
#                 ctx.specular_reflectance = scene.param_map[key].bsdf.specular_reflectance.data
#         scene.configure()
        
#         ctx.masks = [integrator_mask.renderD(scene, id) for id in sensor_ids]
#         uvs = [integrator_uv.renderD(scene, id) for id in sensor_ids]

#         if bsdf_type == psdr_cuda.DiffuseBSDF:
#             masks = [ctx.masks[i] > 0. for i in range(len(ctx.masks))]
#             ctx.alb_imgs = [ek.select(masks[i],
#                             scene.param_map[key].bsdf.reflectance.eval(Vector2fD(uvs[i][0], uvs[i][1])), 0)
#                             for i in range(len(uvs))]
#         else:
#             raise NotImplementedError('Not implemented for specular bsdf currently')

#         alb_imgs = torch.stack([alb.torch() for alb in ctx.alb_imgs])
        
#         # ek.cuda_malloc_trim()
#         return alb_imgs

#     @staticmethod
#     def backward(ctx, grad_out):
#         for i in range(len(ctx.imgs)):
#             ek.set_gradient(ctx.alb_imgs[i], Vector3fC(grad_out[i]))
#         FloatD.backward()

#         bsdf_type = type(config.scene.param_map[config.key].bsdf)

#         grad_v = ek.gradient(ctx.v)
#         nan_mask = ek.isnan(grad_v)
#         grad_v = ek.select(nan_mask, 0, grad_v).torch()

#         grad_env = ek.gradient(ctx.env)
#         nan_mask = ek.isnan(grad_env)
#         grad_env = ek.select(nan_mask, 0, grad_env).torch()

#         if bsdf_type == psdr_cuda.DiffuseBSDF:
#             grad_albedo = ek.gradient(ctx.albedo)
#             nan_mask = ek.isnan(grad_albedo)
#             grad_albedo = ek.select(nan_mask, 0, grad_albedo).torch()
#         else:
#             grad_albedo = ek.gradient(ctx.specular_reflectance)
#             nan_mask = ek.isnan(grad_albedo)
#             grad_albedo = ek.select(nan_mask, 0, grad_albedo).torch()

#         # ek.cuda_malloc_trim()
#         result = grad_v, grad_albedo, grad_env, None

#         max, min = torch.max(result[1]), torch.min(result[1])
#         if max > 1000 or min < -1000:
#             print(max, min)
#         # msg = ''
#         # if torch.isinf(max):
#         #     msg += 'inf' + ' '
#         # if torch.isinf(min):
#         #     msg += '-inf'
#         # if msg != '': print(msg) 
        
#         result[1].clamp_(-10000, 10000)

#         del ctx.v, ctx.env, ctx.imgs, ctx.masks, ctx.alb_imgs, ctx.nrm_imgs
#         if bsdf_type == psdr_cuda.DiffuseBSDF:
#             del ctx.albedo
#         else:
#             del ctx.alpha_u, ctx.alpha_v, ctx.eta, ctx.k, ctx.specular_reflectance

#         return result
# renderD_alb = RenderD_Albedo.apply