import sys, hashlib
import numpy as np
from scipy.ndimage import gaussian_filter

def build(pools, donor, family, centers, rng, excluded=None):
    from texture_base import transformed, fiber_map, support
    from texture_objects import construct_map
    blocked = set(excluded or [])
    selected = []
    maps = []
    classes = []
    count = int(rng.integers(2, 5))
    for j in range(count):
        cls = int(rng.integers(5))
        options = [(sid, image) for sid, image in pools[cls] if sid not in blocked]
        if not options:
            raise RuntimeError('Independent own-source templates unavailable')
        sid, image = options[int(rng.integers(len(options)))]
        blocked.add(sid)
        source = [[] for _ in range(5)]
        source[cls] = [(sid, image)]
        if family == 'natural':
            value = transformed(image, rng)
        elif family == 'spots':
            value, _ = construct_map(source, cls, str(rng.choice(['puncta', 'cluster'])), centers, 0.5, rng)
        elif family == 'fibers':
            value, _ = fiber_map(source, cls, rng)
        elif family == 'diffuse':
            v = transformed(image, rng)
            value = 0.7 * gaussian_filter(v, float(rng.uniform(1, 12)), mode='reflect') + 0.3 * v
        else:
            raise RuntimeError('Declared texture family')
        rms = float(np.sqrt(np.mean(value.astype(float) ** 2)))
        assert np.isfinite(value).all() and value.min() >= 0 and (rms > 0)
        maps.append(value / rms)
        selected.append(sid)
        classes.append(cls)
    yy, xx = np.mgrid[:256, :256]
    fields = []
    for j in range(count):
        cy, cx = rng.uniform(-40, 296, 2)
        sy, sx = rng.uniform(45, 210, 2)
        fields.append(np.exp(-0.5 * ((yy - cy) / sy) ** 2 - 0.5 * ((xx - cx) / sx) ** 2) + 0.03)
    fields = np.stack(fields)
    fields /= fields.sum(0, keepdims=True)
    gains = np.exp(rng.uniform(np.log(0.3), 0, count))[:, None, None]
    value = np.sum(np.stack(maps) * fields * gains, axis=0)
    masked = bool(rng.random() < 0.3)
    shift = [0, 0]
    if masked:
        peak = np.array(np.unravel_index(int(value.argmax()), value.shape))
        target = np.rint(centers[np.argmin(np.sum((centers - peak) ** 2, axis=1))]).astype(int)
        target = np.clip(target, 0, 255)
        dy, dx = (target - peak).tolist()
        shift = [dy, dx]
        placed = np.zeros_like(value)
        sy0, sx0 = (max(0, -dy), max(0, -dx))
        sy1, sx1 = (min(256, 256 - dy), min(256, 256 - dx))
        placed[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = value[sy0:sy1, sx0:sx1]
        value = placed * support(centers, rng)
    rms = float(np.sqrt(np.mean(value.astype(float) ** 2)))
    assert np.isfinite(value).all() and value.min() >= 0 and (rms > 0)
    value = np.ascontiguousarray(value / rms, dtype=np.float32)
    return (value, {'source': selected, 'source_classes': classes, 'family': family, 'source_count': count, 'nuclear_special_case': False, 'elliptical_support': masked, 'source_global_amplitude_transferred': False, 'placement_shift': shift, 'texture_SHA256': hashlib.sha256(value.tobytes()).hexdigest()})
