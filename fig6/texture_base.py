import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from texture_objects import construct_map, patch, resize, source, stamp, window
FAMILIES = ('natural', 'spots', 'fibers', 'diffuse')

def select_source(pools, donor, rng, excluded=None):
    available = [i for i, (sid, _) in enumerate(pools[donor]) if sid != excluded]
    if not available:
        raise RuntimeError('Independent own-source prior required')
    sid, image = pools[donor][available[int(rng.integers(len(available)))]]
    selected = [[] for _ in range(5)]
    selected[donor] = [(sid, image)]
    return (sid, image, selected)

def transformed(image, rng):
    image = np.rot90(image, int(rng.integers(4)))
    if rng.random() < 0.5:
        image = image[:, ::-1]
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    angle = float(rng.uniform(-np.pi, np.pi))
    sy, sx = np.exp(rng.uniform(np.log(0.6), np.log(1.7), 2))
    y = (np.cos(angle) * (yy - 127.5) + np.sin(angle) * (xx - 127.5)) / sy + 127.5
    x = (-np.sin(angle) * (yy - 127.5) + np.cos(angle) * (xx - 127.5)) / sx + 127.5
    elastic = float(rng.uniform(0, 8))
    for coordinate in (y, x):
        displacement = gaussian_filter(rng.standard_normal((256, 256)).astype(np.float32), 18.0, mode='reflect')
        coordinate += elastic * displacement / max(float(displacement.std()), 1e-12)
    value = 0.95 * np.maximum(map_coordinates(image, [y, x], order=1, mode='reflect', prefilter=False), 0) + 0.05 * image
    positive = value[value > 0]
    if not positive.size:
        raise RuntimeError('Own source has no positive support')
    anchor = max(float(np.percentile(positive, 95)), 1e-12)
    return np.asarray(anchor * (value / anchor) ** float(rng.uniform(0.7, 1.4)), np.float32)

def support(centers, rng):
    yy, xx = np.mgrid[:256, :256]
    value = np.zeros((256, 256), np.float32)
    for cy, cx in centers:
        ry, rx = rng.uniform(25, 110, 2)
        angle = float(rng.uniform(0, np.pi))
        dy, dx = (yy - cy, xx - cx)
        radius = np.sqrt(((np.cos(angle) * dy + np.sin(angle) * dx) / ry) ** 2 + ((-np.sin(angle) * dy + np.cos(angle) * dx) / rx) ** 2)
        value = np.maximum(value, np.where(radius < 1, 0.5 + 0.5 * np.cos(np.pi * np.minimum(radius, 1)), 0) ** 2)
    return np.asarray(value, np.float32)

def fiber_map(selected, donor, rng):
    sid, image, coordinates = source(selected, donor, rng)
    value = np.zeros((256, 256), np.float32)
    paths = []
    for _ in range(int(rng.integers(4, 19))):
        controls = rng.uniform(12, 244, (3, 2))
        width = int(rng.integers(3, 12))
        kernel = window(resize(patch(image, coordinates, 16, rng), (width, width)), 0.85)
        t = np.linspace(0, 1, int(rng.integers(16, 49)))[:, None]
        points = (1 - t) ** 2 * controls[0] + 2 * (1 - t) * t * controls[1] + t * t * controls[2]
        for cy, cx in points:
            stamp(value, kernel, cy, cx)
        paths.append(controls.tolist())
    return (value, {'source': sid, 'curved_paths': len(paths), 'geometry_sha256': hashlib.sha256(json.dumps(paths).encode()).hexdigest()})

def build(pools, donor, family, centers, rng, excluded=None):
    sid, image, selected = select_source(pools, donor, rng, excluded)
    if donor == 1 and family != 'natural':
        value, detail = construct_map(selected, donor, 'nucleus', centers, 0.0, rng)
        if family == 'diffuse':
            value = gaussian_filter(value, float(rng.uniform(0.5, 2.5)), mode='reflect')
    elif family == 'natural':
        value, detail = (transformed(image, rng), {'source': sid, 'natural_layout_retained': True})
    elif family == 'spots':
        value, detail = construct_map(selected, donor, str(rng.choice(['puncta', 'cluster'])), centers, 0.5, rng)
    elif family == 'fibers':
        value, detail = fiber_map(selected, donor, rng)
    elif family == 'diffuse':
        natural = transformed(image, rng)
        sigma = float(rng.uniform(1, 12))
        value = 0.7 * gaussian_filter(natural, sigma, mode='reflect') + 0.3 * natural
        detail = {'source': sid, 'coarse_sigma': sigma}
    else:
        raise RuntimeError('Declared texture family')
    mask = support(centers, rng)
    anchor = max(float(np.mean(value)), float(np.mean(image)), 1e-12)
    value = mask * (value + 0.02 * anchor)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2)))
    if not np.isfinite(value).all() or np.min(value) < 0 or rms <= 0:
        raise RuntimeError('Finite nonempty independently placed texture')
    value = np.asarray(value / rms, np.float32)
    return (value, {'source': sid, 'donor': int(donor), 'family': family, 'nuclear_source_only': donor == 1, 'detail': detail, 'support_fraction': float(np.mean(mask > 0)), 'centers_sha256': hashlib.sha256(centers.tobytes()).hexdigest()})

def latents(config, pools, rng, index, regime=None):
    forced = regime if regime in FAMILIES else None
    profile = regime or ('weak620' if index % 4 == 0 else 'weak_nucleus' if index % 4 == 1 else 'ordinary')
    layout = regime if regime in ('separated', 'overlap') else str(rng.choice(['separated', 'independent', 'overlap']))
    common = rng.uniform(48, 208, (int(rng.integers(1, 5)), 2))
    donors = [1] + [int(v) for v in rng.permutation([0, 2, 3, 4])]
    exposure = float(np.exp(rng.uniform(np.log(0.03), np.log(1.5))))
    desired = exposure * np.exp(rng.uniform(np.log(0.1), 0, 5))
    weak_slot = 3 if profile == 'weak620' else 0 if profile == 'weak_nucleus' else None
    requested_ratio = None
    if weak_slot is not None:
        requested_ratio = float(np.exp(rng.uniform(np.log(0.01), np.log(0.1))))
        desired[weak_slot] = requested_ratio * max((desired[k] for k in range(5) if k != weak_slot))
    components, priors, records = ([], [], [])
    for slot, donor in enumerate(donors):
        family = forced or str(rng.choice(FAMILIES))
        if layout == 'overlap':
            centers = np.clip(common + rng.normal(0, 12, common.shape), 20, 236)
        elif layout == 'separated':
            center = np.array([[42 + int(slot // 3) * 145, 37 + int(slot % 3) * 88]], dtype=float)
            centers = np.clip(center + rng.normal(0, 9, center.shape), 20, 236)
        else:
            centers = rng.uniform(24, 232, (int(rng.integers(1, 5)), 2))
        image, metadata = build(pools, donor, family, centers, rng)
        prior_centers = rng.uniform(24, 232, (len(centers), 2))
        prior, prior_metadata = build(pools, donor, family, prior_centers, rng, metadata['source'])
        components.append(image * desired[slot])
        priors.append(prior * desired[slot])
        records.append({'slot': slot, 'family': family, 'texture': metadata, 'prior_texture': prior_metadata, 'requested_preH_RMS': float(desired[slot])})
    components, priors = (np.stack(components), np.stack(priors))
    gain = min(1.0, 8.0 / max(float(components.max()), float(priors.max())))
    components *= gain
    priors *= gain
    bg_donor = int(rng.choice([0, 2, 3, 4]))
    _, image, _ = select_source(pools, bg_donor, rng)
    field = gaussian_filter(image, float(rng.uniform(8, 24)), mode='reflect')
    field = np.ones_like(field, dtype=np.float32)
    calibration = config['_calibration']
    amplitude = float(rng.choice(calibration['background_peak_DN_scales'])) * float(np.exp(rng.uniform(np.log(0.5), np.log(2)))) / config['DN_unit']
    reference = np.concatenate([components, (field * amplitude)[None]], axis=0).astype(np.float32)
    matrix = np.asarray(calibration['matrix8x6'], np.float32)
    clean = np.einsum('mk,khw->mhw', matrix, reference, optimize=True).astype(np.float32)
    detector_gain = min(1.0, 62000.0 / max(float(clean.max()) * config['DN_unit'], 1e-12))
    reference *= detector_gain
    priors *= detector_gain
    clean *= detector_gain
    metadata = {'index': int(index), 'profile': profile, 'layout': layout, 'donors': donors, 'texture_modes': [r['family'] for r in records], 'sources': records, 'weak_slot': weak_slot, 'requested_weak_ratio': requested_ratio, 'preH_common_peak_gain': gain, 'preH_detector_gain': detector_gain, 'realized_preH_RMS': np.sqrt(np.mean(reference[:5].astype(np.float64) ** 2, axis=(1, 2))).tolist(), 'nucleus_always_DAPI': True, 'external_texture': False, 'component_GT_in_loss': False, 'all_five_globally_present': True, 'noise_order': 'after common H, in batch constructor', 'background_amplitude': amplitude, 'background_spatially_constant': True}
    json.dumps(metadata, allow_nan=False)
    return (reference, clean, priors.astype(np.float32), metadata)
