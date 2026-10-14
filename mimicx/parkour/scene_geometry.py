"""CPU geometry helpers for fixed, metrically recorded scene fixtures."""
import numpy as np


def descending_staircase(end, tread=.42, rise=.085, width=1.8):
    """Robot-scale descending support, with ground beyond the final riser."""
    if not np.isfinite([end, tread, rise, width]).all() or min(end, tread, rise, width) <= 0:
        raise ValueError('Descending staircase dimensions must be positive and finite')
    count = max(1, int(np.ceil(end / tread)))
    edges = np.linspace(0., end, count + 1)
    edges[0] = -1.
    boxes = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        top = (count - i) * rise
        boxes.append(dict(name=f'down_step_{i:02d}', center=[(a+b)/2, 0., top/2],
                          half_size=[(b-a)/2, width/2, top/2], yaw=0., friction=.8,
                          rgba=[.54, .60, .57, 1.]))
    return boxes


def inset_tread_position(x, box, margin=.08):
    half = box['half_size'][0]
    if not 0 <= margin < half:
        raise ValueError('Foot inset must be smaller than the tread half-depth')
    center = box['center'][0]
    return float(np.clip(x, center-half+margin, center+half-margin))


def box_signed_distance(points, center, half_size, yaw):
    p = np.asarray(points)-np.asarray(center)
    c, s = np.cos(yaw), np.sin(yaw)
    local = p @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
    q = np.abs(local)-np.asarray(half_size)
    return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)


def validate_scene(scene):
    if not scene.get('boxes'):
        raise ValueError('Scene must contain fixed collision boxes')
    names = set()
    for box in scene['boxes']:
        if box['name'] in names:
            raise ValueError('Duplicate collision geometry name')
        names.add(box['name'])
        for key in ('center','half_size'):
            x = np.asarray(box[key],dtype=float)
            if x.shape != (3,) or not np.isfinite(x).all():
                raise ValueError(f'Invalid {key}')
        if np.any(np.asarray(box['half_size']) <= 0):
            raise ValueError('Box half sizes must be positive')
        if not np.isfinite(box['yaw']) or not 0 < box['friction'] <= 3:
            raise ValueError('Invalid orientation or friction')
