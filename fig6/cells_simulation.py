import numpy as np

def hmap(a, sigma, factor):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    if sigma:
        a = gaussian_filter(a, (0, sigma, sigma), mode='reflect')
    if factor > 1:
        a = a.reshape(len(a), 256 // factor, factor, 256 // factor, factor).mean((2, 4)).repeat(factor, 1).repeat(factor, 2)
    return np.ascontiguousarray(a.reshape(len(a), 128, 2, 128, 2).mean((2, 4)), dtype=np.float32)

def generate_cells(c, pool, assets, split, index, profile=None, target=None):
    import numpy as np
    from texture_mix import build
    rng = np.random.default_rng(np.random.SeedSequence([c['seed'], {'train': 11, 'val': 21, 'test': 31, 'preflight': 41}[split], index]))
    if profile is None:
        profile = str(rng.choice(['ordinary', 'weak', 'single', 'pair', 'absent', 'blank', 'overlap'], p=[0.3, 0.25, 0.1, 0.1, 0.1, 0.05, 0.1]))
    target = index % 5 if target is None else target
    active = list(range(5))
    if profile == 'single':
        active = [target]
    elif profile == 'pair':
        active = [target, (target + 1 + index % 4) % 5]
    elif profile == 'absent':
        active = [k for k in range(5) if k != target]
    elif profile == 'blank':
        active = []
    exposure = float(np.exp(rng.uniform(np.log(2.0), np.log(300.0))))
    strength = exposure * np.exp(rng.uniform(np.log(0.05), 0, 5))
    if profile == 'weak':
        strength[target] = max(strength) * float(np.exp(rng.uniform(np.log(0.01), np.log(0.1))))
    refs = []
    priors = []
    textures = []
    shared = rng.uniform(45, 211, (int(rng.integers(1, 5)), 2))
    for k in range(5):
        family = str(rng.choice(['natural', 'spots', 'fibers', 'diffuse']))
        centers = np.clip(shared + rng.normal(0, 10, shared.shape), 20, 236) if profile == 'overlap' else rng.uniform(24, 232, (int(rng.integers(1, 5)), 2))
        a, m = build(pool, 0, family, centers, rng)
        p, n = build(pool, 0, family, rng.uniform(24, 232, (len(centers), 2)), rng, m['source'])
        refs.append(a * strength[k] if k in active else np.zeros_like(a))
        priors.append(p * strength[k] if k in active else np.zeros_like(p))
        textures.append({'family': family, 'sources': m['source'], 'prior_sources': n['source']})
    u = rng.random()
    sigma = float(rng.uniform(0.25, 2.5)) if 0.2 <= u < 0.55 or u >= 0.75 else 0.0
    factor = int(rng.choice([2, 4, 8])) if u >= 0.55 else 1
    x = hmap(np.stack(refs), sigma, factor)
    p = hmap(np.stack(priors), sigma, factor)
    clean = np.einsum('ck,khw->chw', assets['matrix'], x, optimize=True)
    gain = min(1.0, 220 / max(float(clean.max()), 1e-20))
    x *= gain
    p *= gain
    clean *= gain
    sig = np.full(28, float(np.exp(rng.uniform(np.log(0.25), np.log(2.0)))), np.float32)
    alpha = float(rng.choice([0.0, 0.5, 1.0, 2.0]))
    variance = sig[:, None, None] ** 2 + alpha * clean
    y = np.floor(np.clip(clean + rng.normal(size=clean.shape) * np.sqrt(variance), 0, 255) + 0.5).astype(np.uint8)
    meta = {'split': split, 'index': index, 'profile': profile, 'active': active, 'weak': target if profile == 'weak' else None, 'textures': textures, 'effective_shot_alpha_DN': alpha, 'gain_common': gain, 'H': {'sigma': sigma, 'factor': factor, 'same_FOV_area_factor': 2, 'post_H_RMS_restore': False}}
    return (y, p.astype(np.float32), x.astype(np.float32), sig, alpha, meta)

def inputs_from_raw(raw, c, branch):
    import numpy as np
    if branch == 'A':
        return np.stack([raw[g].sum(0, dtype=np.float32) for g in c['A_groups']])
    assert branch == 'B'
    return raw[c['B']].astype(np.float32)

def independent_inputs(c, ref, matrix, sig, alpha, split, index):
    import numpy as np
    clean = np.einsum('ck,khw->chw', matrix, ref, optimize=True)
    sd = np.sqrt(sig[:, None, None] ** 2 + alpha * clean)
    views = []
    seeds = []
    for stream, branch in [(901, 'A'), (902, 'B')]:
        seed = [c['seed'], stream, {'train': 11, 'val': 21, 'test': 31, 'preflight': 41}[split], index]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        noisy = np.floor(np.clip(clean + rng.standard_normal(clean.shape) * sd, 0, 255) + 0.5).astype(np.uint8)
        v = inputs_from_raw(noisy, c, branch)
        assert np.isfinite(v).all() and v.min() >= 0 and (v.max() <= 2040)
        views.append(v.astype(np.uint16 if branch == 'A' else np.uint8))
        seeds.append(seed)
    return (views, seeds)
