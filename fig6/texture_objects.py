import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter, rotate, zoom
_coordinates = {}

def source(pools, donor, rng):
    sid, image = pools[donor][int(rng.integers(len(pools[donor])))]
    key = id(image)
    if key not in _coordinates:
        positive = image[image > 0]
        if not positive.size:
            raise RuntimeError('Positive own-source feature required')
        coordinates = np.argwhere(image >= np.percentile(positive, 80))
        _coordinates[key] = coordinates[::max(1, len(coordinates) // 2048)]
    return (sid, image, _coordinates[key])

def patch(image, coordinates, size, rng):
    row, col = coordinates[int(rng.integers(len(coordinates)))]
    half = size // 2
    row, col = np.clip([row, col], half, np.array(image.shape) - half - 1)
    value = image[row - half:row + half + 1, col - half:col + half + 1].copy()
    value = np.maximum(value, 0)
    if not np.any(value > 0):
        raise RuntimeError('Selected own-source feature has no contrast')
    return value

def resize(value, shape):
    sigma = tuple((max(0.0, 0.5 * (value.shape[k] / shape[k] - 1)) for k in range(2)))
    if max(sigma) > 0:
        value = gaussian_filter(value, sigma, mode='reflect')
    output = zoom(value, (shape[0] / value.shape[0], shape[1] / value.shape[1]), order=1, mode='nearest', prefilter=False)
    return np.maximum(output, 0).astype(np.float32)

def window(value, width=0.7):
    yy, xx = np.mgrid[-1:1:complex(value.shape[0]), -1:1:complex(value.shape[1])]
    radius = np.sqrt((xx / width) ** 2 + (yy / width) ** 2)
    envelope = np.where(radius < 1, 0.5 + 0.5 * np.cos(np.pi * np.minimum(radius, 1)), 0) ** 2
    value = value * envelope
    value[[0, -1], :] = 0
    value[:, [0, -1]] = 0
    peak = float(value.max())
    if not np.isfinite(value).all() or peak <= 0:
        raise RuntimeError('Finite positive own-source object')
    return np.asarray(value / peak, np.float32)

def stamp(canvas, kernel, row, col):
    top, left = (int(round(row)) - kernel.shape[0] // 2, int(round(col)) - kernel.shape[1] // 2)
    bottom, right = (top + kernel.shape[0], left + kernel.shape[1])
    y0, x0, y1, x1 = (max(0, top), max(0, left), min(canvas.shape[0], bottom), min(canvas.shape[1], right))
    if y0 < y1 and x0 < x1:
        canvas[y0:y1, x0:x1] += kernel[y0 - top:y1 - top, x0 - left:x1 - left]

def positive_gain(image, desired):
    rms = float(np.sqrt(np.mean(image.astype(np.float64) ** 2)))
    if rms <= 0:
        raise RuntimeError('Globally present synthetic component required')
    gain = min(desired / rms, 8.0 / float(image.max()))
    value = np.asarray(image * gain, np.float32)
    return (value, float(np.sqrt(np.mean(value.astype(np.float64) ** 2))))

def construct_map(pools, donor, family, centers, coupling, rng, size=256):
    sid, image, coordinates = source(pools, donor, rng)
    output = np.zeros((size, size), np.float32)
    placements = []
    if family == 'nucleus':
        for center in centers:
            value = patch(image, coordinates, int(rng.choice([64, 96, 128, 160])), rng)
            side = int(rng.integers(48, 177))
            kernel = window(rotate(resize(value, (side, side)), float(rng.uniform(0, 360)), reshape=True, order=1, mode='constant', cval=0, prefilter=False), 0.85)
            stamp(output, kernel, *center)
        count = len(centers)
    else:
        count = int(rng.integers(12, 97)) if family == 'puncta' else int(rng.integers(3, 17)) if family == 'cluster' else int(rng.integers(4, 25)) if family == 'short_line' else int(rng.integers(2, 9))
        for number in range(count):
            center = np.asarray(centers[int(rng.integers(len(centers)))]) + rng.normal(0, 20 if coupling == 1 else 45, 2) if rng.random() < coupling else rng.uniform(8, size - 8, 2)
            center = np.clip(center, 4, size - 5)
            if family in ['puncta', 'cluster']:
                for spot in range(1 if family == 'puncta' else int(rng.integers(3, 11))):
                    side = 2 * int(rng.integers(1, 8)) + 1
                    value = patch(image, coordinates, 12, rng)
                    kernel = window(resize(value, (side, side)), float(rng.uniform(0.3, 0.8)))
                    point = center + (rng.normal(0, float(rng.uniform(2, 10)), 2) if family == 'cluster' else 0)
                    stamp(output, kernel * float(np.exp(rng.uniform(np.log(0.2), 0))), *point)
            elif family == 'short_line':
                value = patch(image, coordinates, 32, rng)
                kernel = window(resize(value, (int(rng.integers(3, 10)), int(rng.integers(8, 37)))))
                kernel = rotate(kernel, float(rng.uniform(0, 180)), reshape=True, order=1, mode='constant', cval=0, prefilter=False)
                stamp(output, kernel, *center)
            elif family == 'diffuse':
                value = gaussian_filter(patch(image, coordinates, 64, rng), float(rng.uniform(2, 8)), mode='reflect')
                side = int(rng.integers(31, 97))
                stamp(output, window(resize(value, (side, side))), *center)
            else:
                raise RuntimeError('Declared morphology family')
            placements.append([float(center[0]), float(center[1])])
    if not np.isfinite(output).all() or not np.any(output > 0):
        raise RuntimeError('Nonempty finite recomposed morphology')
    return (output, {'source': sid, 'donor': int(donor), 'family': family, 'objects_or_clusters': count, 'original_global_layout_retained': False, 'placement_SHA256': hashlib.sha256(json.dumps(placements).encode()).hexdigest()})

def latents(config, pools, rng, index, regime=None):
    centers = rng.uniform(32, 224, (int(rng.integers(1, 5)), 2))
    coupling = 1.0 if regime == 'nuclear_confounded' else 0.6 if rng.random() < 0.3 else 0.0
    donors = [1] + [int(v) for v in rng.permutation([0, 2, 3, 4])]
    families = ['nucleus']
    for slot in range(4):
        family = str(rng.choice(['puncta', 'cluster', 'short_line', 'diffuse'], p=[0.4, 0.25, 0.15, 0.2]))
        if regime == 'puncta':
            family = 'puncta'
        if regime == 'cluster_line':
            family = 'cluster' if slot % 2 == 0 else 'short_line'
        if regime == 'diffuse':
            family = 'diffuse'
        families.append(family)
    profile = regime or ('weak620' if (index - 1) % 4 == 0 else 'weak_nucleus' if (index - 1) % 8 == 1 else 'mixed')
    nuclear = float(np.exp(rng.uniform(np.log(0.02), np.log(2))))
    exposure = float(np.exp(rng.uniform(np.log(0.03), np.log(1.5))))
    if profile == 'weak_nucleus':
        nuclear = float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))
    components, priors, records = ([], [], [])
    for slot, (donor, family) in enumerate(zip(donors, families)):
        image, metadata = construct_map(pools, donor, family, centers, coupling, rng)
        desired = nuclear if slot == 0 else exposure * float(np.exp(rng.uniform(np.log(0.05), 0)))
        if slot == 3 and profile == 'weak620':
            desired = float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))
        value, realized = positive_gain(image, desired)
        prior_centers = rng.uniform(32, 224, (int(rng.integers(1, 5)), 2))
        prior_image, prior_metadata = construct_map(pools, donor, family, prior_centers, coupling, rng)
        prior, prior_realized = positive_gain(prior_image, realized)
        components.append(value)
        priors.append(prior)
        records.append({'slot': slot, **metadata, 'desired_RMS': desired, 'realized_preH_RMS': realized, 'prior': prior_metadata, 'prior_realized_preH_RMS': prior_realized})
    bg_donor = int(rng.choice([0, 2, 3, 4]))
    field, bgmeta = construct_map(pools, bg_donor, 'diffuse', centers, 0.0, rng)
    field = gaussian_filter(field, float(rng.uniform(8, 24)), mode='reflect')
    field = 0.5 + 0.5 * field / max(float(field.mean()), 1e-12)
    scales = config['_calibration']['background_peak_DN_scales']
    amplitude = float(rng.choice(scales)) * float(np.exp(rng.uniform(np.log(0.5), np.log(2)))) / config['DN_unit']
    components.append(np.asarray(field * amplitude, np.float32))
    reference = np.stack(components)
    calibration = config['_calibration']
    matrix = np.asarray(calibration['matrix8x6'], np.float32)
    clean = np.einsum('mk,khw->mhw', matrix, reference, optimize=True).astype(np.float32)
    sigma = np.asarray(calibration['noise32_MAD_proxy_DN'], np.float32)[config['indices32']] / config['DN_unit']
    noise_factor = float(rng.uniform(0.5, 1.5))
    dependent = bool(rng.random() < 0.5)
    variance = np.sqrt(0.5 + 0.5 * np.clip(clean / np.maximum(clean.mean((1, 2), keepdims=True), 1e-08), 0, 4)) if dependent else 1.0
    observed = clean + rng.standard_normal(clean.shape, dtype=np.float32) * sigma[:, None, None] * noise_factor * variance
    metadata = {'index': int(index), 'profile': profile, 'donors': donors, 'families': families, 'nucleus_always_DAPI': True, 'coupling_probability': coupling, 'sources': records, 'background': bgmeta, 'background_amplitude': amplitude, 'noise_factor': noise_factor, 'signal_dependent_MAD_proxy': dependent, 'external_texture': False, 'component_GT_in_loss': False, 'all_five_globally_present': True}
    json.dumps(metadata, allow_nan=False)
    return (reference, np.asarray(observed, np.float32), np.stack(priors), metadata)
