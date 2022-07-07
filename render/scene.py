import psdr_cuda
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector2f as Vector2fD, Vector3f as Vector3fD, Vector3i as Vector3iD
import numpy as np
from .util import transform, flip_y, flip_y_np, wrap_np
from . import config
from .renderD import renderD, renderD_demod

class Scene:
    def __init__(self, xml):
        self._scene = psdr_cuda.Scene()
        self._scene.load_string(xml, False)

        self._integrator = psdr_cuda.DirectIntegrator()
        self._integrator.hide_emitters = True
        self._integrator_demod = psdr_cuda.DirectLightingIntegrator()
        self._integrator_demod.hide_emitters = True
        self._integrator_mask = psdr_cuda.FieldExtractionIntegrator('silhouette')
        self._integrator_uv = psdr_cuda.FieldExtractionIntegrator('uv')
        self._integrator_alb = psdr_cuda.FieldExtractionIntegrator('albedo')
        self._integrator_depth = psdr_cuda.FieldExtractionIntegrator('depth')
        self._integrator_normal = psdr_cuda.FieldExtractionIntegrator('shNormal')

        self._configured = False

        self.num_sensors = self._scene.num_sensors

    def _set_config(self, key, sensor_ids):
        config.scene = self._scene
        config.key = key
        config.integrator = self._integrator
        config.integrator_demod = self._integrator_demod
        config.integrator_mask = self._integrator_mask
        config.integrator_uv = self._integrator_uv
        config.integrator_alb = self._integrator_alb
        config.integrator_depth = self._integrator_depth
        config.integrator_normal = self._integrator_normal
        config.sensor_ids = sensor_ids

    def set_opts(self, res, spp, sppe = None, sppse = None, log_level=0):
        self._scene.opts.width = res[0]
        self._scene.opts.height = res[1]
        self._scene.opts.spp = spp
        self._scene.opts.log_level = log_level
        if sppe is not None:
            self._scene.opts.sppe = sppe
        if sppse is not None:
            self._scene.opts.sppse = sppse
        self._configured = False

    def renderC(self, sensor_ids = None, img_type='shaded'):
        if self._configured == False:
            self._scene.configure()
            self._configured = True
        if sensor_ids is None: sensor_ids = range(self.num_sensors)

        if img_type == 'shaded':
            integrator = self._integrator
        elif img_type == 'demodulated':
            integrator = self._integrator_demod
        elif img_type == 'mask':
            integrator = self._integrator_mask
        elif img_type == 'uv':
            integrator = self._integrator_uv
        elif img_type == 'albedo':
            integrator = self._integrator_alb
        elif img_type == 'normal':
            integrator = self._integrator_normal
        elif img_type == 'depth':
            integrator = self._integrator_depth
        else:
            raise NotImplementedError('Unknown image type')

        imgs = []
        for id in sensor_ids:
            imgs.append(integrator.renderC(self._scene, id).numpy())
            ek.cuda_malloc_trim()
        return imgs

    def renderD(self, v, mat, env, key, sensor_ids):
        # if self._scene.opts.sppse > 0:
        #     self._scene.configure()
        #     for id in sensor_ids:
        #         self._integrator.preprocess_secondary_edges(self._scene, id,
        #                         np.array([10000, 5, 5, 2]), 16)

        self._set_config(key, sensor_ids)
        imgs = renderD(v, mat, env)
        self._configured = True
        
        return imgs

    def renderD_demod(self, v, mat, env, key, sensor_ids):
        # if self._scene.opts.sppse > 0:
        #     self._scene.configure()
        #     for id in sensor_ids:
        #         self._integrator_demod.preprocess_secondary_edges(self._scene, id,
        #                         np.array([10000, 5, 5, 2]), 16)

        self._set_config(key, sensor_ids)
        imgs = renderD_demod(v, mat, env)
        self._configured = True

        return imgs

    def reload_mesh(self, key, v, f, uv = None, uv_idx = None):
        m = self._scene.param_map[key]
        v_ = Vector3fD(v)
        f_ = Vector3iD(f)
        # psdr-cuda internally stores uvs in the OpenGL coordinate convention, i.e. (0,0) is at the bottom left
        # while our program uses the scanline convention, i.e. (0,0) is at the top left
        uv_ = Vector2fD() if uv is None else Vector2fD(flip_y(uv))
        uv_idx_ = Vector3iD() if uv_idx is None else Vector3iD(uv_idx)
        self._scene.reload_mesh_mem(m, v_, f_, uv_, uv_idx_)
        self._configured = False
    
    def reload_mat(self, key, mat=None, res=None, res_only=False):
        assert(not (mat is None and res is None))

        m = self._scene.param_map[key]
        if res is None:
            res = mat.shape[:-1]
            assert(len(res) == 1 or len(res) == 2)
            if len(res) == 1:
                # when mat stores vertex attributes
                res = (1, res[0])
            else:
                # flip to make res = (w, h)
                res = (res[1], res[0])

        bsdf_type = type(m.bsdf)
        if bsdf_type == psdr_cuda.DiffuseBSDF:
            m.bsdf.reflectance.resolution = res
        elif bsdf_type == psdr_cuda.RoughConductorBSDF:
            m.bsdf.alpha_u.resolution = res
            m.bsdf.alpha_v.resolution = res
            m.bsdf.eta.resolution = res
            m.bsdf.k.resolution = res
            m.bsdf.specular_reflectance.resolution = res
        else:
            raise NotImplementedError('Unsupported BSDF: ' + bsdf_type.__name__)
        if not res_only:
            assert(mat is not None)
            if type(m.bsdf) == psdr_cuda.DiffuseBSDF:
                m.bsdf.reflectance.data = Vector3fD(mat.reshape(-1,3))
            elif bsdf_type == psdr_cuda.RoughConductorBSDF:
                m.bsdf.alpha_u.data = FloatD(mat[...,0].reshape(-1))
                m.bsdf.alpha_v.data = FloatD(mat[...,0].reshape(-1))
                m.bsdf.eta.data = Vector3fD(mat[...,1:4].reshape(-1,3))
                m.bsdf.k.data = Vector3fD(mat[...,4:7].reshape(-1,3))
                m.bsdf.specular_reflectance.data = Vector3fD(mat[...,7:].reshape(-1,3))
            else:
                raise NotImplementedError('Unsupported BSDF: ' + bsdf_type.__name__)
    
    def reload_envmap(self, data, res=None):
        if res is None:
            res = data.shape[:2]
            res = (res[1], res[0])
        assert(res[0] == 2*res[1])

        # Assuming the environment map is the first emitter
        envmap = self._scene.param_map['Emitter[0]']
        envmap.radiance.resolution = res
        envmap.radiance.data = Vector3fD(data.reshape(-1,3))

        self._configured = False

    def get_mesh(self, key, to_world=False, return_uv=False):
        m = self._scene.param_map[key]
        v = m.vertex_positions
        v = transform(v, m.to_world).numpy() if to_world else v.numpy()
        f = m.face_indices.numpy()

        if not return_uv:
            return v, f
        else:
            uv = m.vertex_uv.numpy()
            uv_idx = m.face_uv_indices.numpy()
            return v, f, wrap_np(flip_y_np(uv)), uv_idx

    def get_mat(self, key):
        bsdf = self._scene.param_map[key].bsdf
        bsdf_type = type(bsdf)
        if bsdf_type == psdr_cuda.DiffuseBSDF:
            mat = bsdf.reflectance.data.numpy()
            res = bsdf.reflectance.resolution
            mat = mat.reshape(res[1], res[0], 3)
        elif bsdf_type == psdr_cuda.RoughConductorBSDF:
            res = bsdf.alpha_u.resolution
            alpha = bsdf.alpha_u.data.numpy()
            alpha = alpha[...,None]
            eta = bsdf.eta.data.numpy()
            k = bsdf.k.data.numpy()
            specular_reflectance = bsdf.specular_reflectance.data.numpy()
            mat = np.concatenate([alpha,eta,k,specular_reflectance], axis=-1)
            mat = mat.reshape(res[1], res[0], -1)
        else:
            raise NotImplementedError('Unsupported BSDF: ' + bsdf_type.__name__)

        return mat

    def get_envmap(self):
        envmap = self._scene.param_map['Emitter[0]']
        res = envmap.radiance.resolution
        envmap_tex = envmap.radiance.data.numpy()

        return envmap_tex.reshape(res[1], res[0], 3)

    def update_vertex_positions(self, key, v):
        m = self._scene.param_map[key]
        m.vertex_positions = Vector3fD(v)
        self._configured = False

    def dump(self, key, fname):
        m = self._scene.param_map[key]
        m.dump(fname)

'''
TODO
    - store multiple psdr scenes to allow for multiple differentiable rendering passes
    - preprocess sencondary edges before renderD
'''