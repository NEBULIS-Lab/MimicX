"""Contact targets for an image-registered wall; no scene refitting from robot feet."""
import numpy as np


def contact_envelope(frame, start, stop, ramp=0):
    if ramp <= 0:
        return float(start <= frame < stop)
    if start <= frame < stop:
        return 1.
    phase = (frame-start+ramp)/ramp if frame < start else (stop+ramp-frame)/ramp
    phase = float(np.clip(phase, 0., 1.))
    return phase*phase*(3-2*phase)


def wall_patch_target(point, box, clearance=.03, margin=.05):
    c = np.asarray(box['center'])
    h = np.asarray(box['half_size'])
    a = box['yaw']
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = (np.asarray(point)[:2] - c[:2]) @ rotation
    limit = np.maximum(h[:2] - margin, 0.)
    xy = np.clip(local, -limit, limit) @ rotation.T + c[:2]
    return np.r_[xy, c[2] + h[2] + clearance]
