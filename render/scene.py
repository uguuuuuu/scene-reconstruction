import psdr_cuda
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector2f as Vector2fD, Vector3f as Vector3fD, Vector3i as Vector3iD
from .util import transform, flip_y, flip_y_np, wrap_np
from . import config
from .renderD import renderDVA, renderDVAM, renderDVAME

class Scene:
    def __init__(self, xml):
        self._scene = psdr_cuda.Scene()
        self._scene.load_string(xml, False)
        self._configured = False

        self.num_sensors = self._scene.num_sensors

    def _set_config(self, key, integrator, integrator_mask, sensor_ids):
        config.scene = self._scene
        config.key = key
        config.integrator = integrator
        config.integrator_mask = integrator_mask
        config.sensor_ids = sensor_ids

    def set_opts(self, res, spp, log_level=0, sppe = None, sppse = None):
        self._scene.opts.width = res[0]
        self._scene.opts.height = res[1]
        self._scene.opts.spp = spp
        self._scene.opts.log_level = log_level
        if sppe is not None:
            self._scene.opts.sppe = sppe
        if sppse is not None:
            self._scene.opts.sppse = sppse
        self._configured = False

    def renderC(self, integrator, sensor_ids = None):
        if self._configured == False:
            self._scene.configure()
            self._configured = True
        if sensor_ids is None: sensor_ids = range(self.num_sensors)

        imgs = []
        for id in sensor_ids:
            imgs.append(integrator.renderC(self._scene, id).numpy())
            ek.cuda_malloc_trim()
        return imgs

    def renderDVA(self, v, key, integrator, integrator_mask, sensor_ids):
        self._set_config(key, integrator, integrator_mask, sensor_ids)
        imgs = renderDVA(v)
        self._configured = True
        return imgs

    def renderDVAM(self, v, mat, key, integrator, integrator_mask, sensor_ids):
        self._set_config(key, integrator, integrator_mask, sensor_ids)
        imgs = renderDVAM(v, mat)
        self._configured = True
        return imgs

    def renderDVAME(self, v, mat, env, key, integrator, integrator_mask, sensor_ids):
        self._set_config(key, integrator, integrator_mask, sensor_ids)
        imgs = renderDVAME(v, mat, env)
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
    
    def reload_mat(self, key, mat, res_only=False):
        m = self._scene.param_map[key]
        res = mat.shape[:-1]
        assert(len(res) == 1 or len(res) == 2)
        if len(res) == 1:
            # when mat stores vertex attributes
            res = (1, res[0])
        else:
            # flip to make res = (w, h)
            res = (res[1], res[0])

        m.bsdf.reflectance.resolution = res
        if not res_only:
            m.bsdf.reflectance.data = Vector3fD(mat.reshape(-1,3))
    
    def reload_envmap(self, data):
        res = data.shape[:2]
        res = (res[1], res[0])
        assert(res[0] == 2*res[1])

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

    def update_vertex_positions(self, key, v):
        m = self._scene.param_map[key]
        m.vertex_positions = Vector3fD(v)
        self._configured = False

    def dump(self, key, fname):
        m = self._scene.param_map[key]
        m.dump(fname)

