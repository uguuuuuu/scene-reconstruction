import psdr_cuda
import enoki as ek
from enoki.cuda_autodiff import Float32 as FloatD, Vector2f as Vector2fD, Vector3f as Vector3fD, Vector3i as Vector3iD
from .util import transform
from . import config
from .renderD import renderDVA

class Scene:
    def __init__(self, xml):
        self._scene = psdr_cuda.Scene()
        self._scene.load_string(xml, False)
        self._configured = False

        self.num_sensors = self._scene.num_sensors

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

    def prepare(self):
        self._scene.configure()
        self._configured = True

    def prepared(self):
        return self._configured

    def renderC(self, integrator, sensor_ids = None):
        if self._configured == False:
            return None

        if sensor_ids is None: sensor_ids = range(self.num_sensors)
        imgs = []
        for id in sensor_ids:
            imgs.append(integrator.renderC(self._scene, id).numpy())
            ek.cuda_malloc_trim()
        return imgs

    def renderDVA(self, v, key, integrator, integrator_mask, sensor_ids):
        config.scene = self._scene
        config.key = key
        config.integrator = integrator
        config.integrator_mask = integrator_mask
        config.sensor_ids = sensor_ids
        imgs =  renderDVA(v)
        self._configured = True
        return imgs

    def reload_mesh(self, key, v, f, uv = None, uv_idx = None):
        m = self._scene.param_map[key]
        v = Vector3fD(v)
        f = Vector3iD(f)
        uv = Vector2fD() if uv is None else Vector2fD(uv)
        uv_idx = Vector3iD() if uv_idx is None else Vector3iD(uv_idx)
        self._scene.reload_mesh_mem(m, v, f, uv, uv_idx)
        self._configured = False

    def get_mesh(self, key, to_world=False):
        m = self._scene.param_map[key]
        v = m.vertex_positions
        v = transform(v, m.to_world).numpy() if to_world else v.numpy()
        f = m.face_indices.numpy()
        return v, f

    def dump(self, key, fname):
        m = self._scene.param_map[key]
        m.dump(fname)

