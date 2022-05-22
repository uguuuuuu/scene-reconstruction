from glob import glob
import pathlib
import time
import xml.etree.ElementTree as ET
import cv2
import imageio.v2 as iio
import torch
import numpy as np
import enoki as ek
from enoki.cuda_autodiff import Vector3f as Vector3fD
from enoki.cuda import Vector3f as Vector3fC
from render.scene import Scene

def linear2srgb(img):
    img[img <= 0.0031308] *= 12.92
    img[img > 0.0031308] = 1.055*img[img > 0.0031308]**(1./2.4) - 0.055
    img = np.uint16(np.clip(img, 0., 1.)*65535)
    return img
def srgb2linear(img):
    img[img <= 0.04045] /= 12.92
    img[img > 0.04045] = ((img[img > 0.04045]+0.055)/1.055)**2.4
    return img
def exr2png(fname):
    img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = linear2srgb(img)
    cv2.imwrite(str(pathlib.Path(fname).with_suffix('.png')), img)
def imgs2video(pattern, dst, fps):
    ext = pathlib.Path(pattern).suffix
    fnames = np.sort(glob(pattern))
    w = iio.get_writer(dst, format='FFMPEG', mode='I', fps=fps)
    if ext == '.exr':
        for fname in fnames:
            img = cv2.imread(fname,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = linear2srgb(img).astype(np.uint8)
            w.append_data(img)
    else:
        for fname in fnames:
            w.append_data(iio.imread(fname))
    w.close()

def save_img(img, fname, res: tuple):
    if type(img) == Vector3fD or type(img) == Vector3fC:
        img = img.numpy()
    elif type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()

    img = img.reshape((res[1], res[0], img.shape[-1]))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pathlib.Path(fname).suffix != '.exr':
        img = linear2srgb(img)
    cv2.imwrite(fname, img)

def xmlfile2str(fname):
    tree = ET.parse(fname)
    return ET.tostring(tree.getroot(), encoding='unicode')

class NotFoundError(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)
def load_tensor(pattern, n_itr = -1):
    '''
    Load a PyTorch tensor

    Parameters
    ----------
    pattern: str
        Pattern of the tensor to load
    n_itr: None or int
        load the latest if None, otherwise load the tensor with the specified number of iteration
    '''
    paths = glob(pattern)
    if len(paths) == 0: return None, 0
    paths = [pathlib.Path(p) for p in paths]

    if n_itr == -1:
        path, itr = paths[0], int(paths[0].suffixes[-2][1:])
        for p in paths:
            itr = int(path.suffixes[-2][1:])
            j = int(p.suffixes[-2][1:])
            if j > itr: path, itr = p, j
        return torch.load(path), itr
    else:
        for p in paths:
            if int(p.suffixes[-2][1:]) == n_itr:
                return torch.load(p), n_itr
        raise NotFoundError('No saved tensors matching the specified number of iteration')

def preprocess_scene(fname, remesh_path=None, sdf_path=None):
    def remove_elem(parent, tag):
        for elem in parent.findall(tag):
            parent.remove(elem)
        for elem in parent:
            remove_elem(elem, tag)
    def find_id(parent, id):
        for elem in parent:
            if elem.get('id') == id:
                return (parent, elem)
        for elem in parent:
            a = find_id(elem, id)
            if a is not None:
                return a
    def replace_filename(elem, fname):
        for string in elem.findall('string'):
            if string.get('name') == 'filename':
                old_fname = string.attrib['value']
                string.attrib['value'] = fname
                return old_fname
    def num_of(parent, tag):
        return len(parent.findall(tag))

    '''
    Construct source scenes
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'target')
    p[0].remove(p[1])
    p, source_shape = find_id(root, 'source')
    source_scene = ET.tostring(root, encoding='unicode')

    source_filename = replace_filename(source_shape, '')
    remesh_scene = None
    if remesh_path is not None:
        replace_filename(source_shape, remesh_path)
        remesh_scene = ET.tostring(root, encoding='unicode')
    sdf_scene = None
    if sdf_path is not None:
        replace_filename(source_shape, sdf_path)
        sdf_scene = ET.tostring(root, encoding='unicode')
    replace_filename(source_shape, source_filename)

    '''
    Construct target scene
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'source')
    p[0].remove(p[1])
    target_scene = ET.tostring(root, encoding='unicode')

    return {
        'src': source_scene,
        'tgt': target_scene,
        'rm': remesh_scene,
        'sdf': sdf_scene,
        'n_sensors': num_of(root, 'sensor')
    }

def flip_uv(uv):
    uv = uv.clone()
    uv[:,1] = 1. - uv[:,1]
    return uv

def renderC_img(xml, integrator, sensor_ids = None, res = (256, 256), spp = 32, load_string=True):
    if load_string == False: xml = xmlfile2str(xml)
    scene = Scene(xml)
    scene.set_opts(res, spp, sppe=0, sppse=0)
    imgs = scene.renderC(integrator, sensor_ids)
    assert(imgs is not None)
    return scene, imgs

class TimerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args) 
class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError(f'Timer is running. Use stop() to stop it')
        self._start_time = time.perf_counter()
    
    def stop(self):
        if self._start_time is None:
            raise TimerError(f'Timer is not running. Use .start() to start it')
        elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed