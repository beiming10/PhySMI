import math
import numpy as np
from scipy.ndimage import gaussian_filter

class SpatialSources:

    def texture(self, pools, rng):
        donor = int(rng.integers(5))
        sid, a = pools[donor][int(rng.integers(len(pools[donor])))]
        cy, cx = rng.integers(0, 129, 2)
        a = a[cy:cy + 128, cx:cx + 128]
        a = np.rot90(a, int(rng.integers(4)))
        a = a[:, ::-1] if rng.random() < 0.5 else a
        scale = max(float(np.percentile(a[a > 0], 95)) if np.any(a > 0) else 0, 1e-08)
        return (sid, 0.5 + 0.5 * np.clip(a / scale, 0, 2))

    def pattern(self, rng, centers, kind):
        p = np.zeros((128, 128), np.float32)
        for cy, cx, angle, half, width in centers:
            dy, dx = (self.yy - cy, self.xx - cx)
            along = dy * np.sin(angle) + dx * np.cos(angle)
            across = dy * np.cos(angle) - dx * np.sin(angle)
            dist = np.sqrt(np.maximum(np.abs(along) - half, 0) ** 2 + across ** 2)
            p = np.maximum(p, np.clip(1 - dist / width, 0, 1) ** 0.5)
        return p

    def draw(self, pools, rng, spec):
        x = np.zeros((13, 128, 128), np.float32)
        origins = []
        active = spec['active']
        kind = spec['kind']
        if not active:
            return (x, origins)
        centers = [(float(rng.uniform(25, 103)), float(rng.uniform(25, 103)), float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(4, 15)), float(rng.uniform(1.5, 5))) for _ in range(int(rng.integers(3, 9)))]
        peak = spec.get('peak', float(np.exp(rng.uniform(math.log(0.5), math.log(100)))))
        for j, k in enumerate(active):
            sid, t = self.texture(pools, rng)
            origins.append(sid)
            if kind == 'separated':
                cc = [(float(rng.uniform(12, 116)), float(rng.uniform(8 + 112 * j / len(active), 8 + 112 * (j + 0.8) / len(active))), a, h, w) for cy, cx, a, h, w in centers]
            elif kind == 'partial' and j:
                cc = [(cy + float(rng.uniform(-5, 5)), cx + float(rng.uniform(-5, 5)), a + float(rng.uniform(-0.4, 0.4)), h, w) for cy, cx, a, h, w in centers]
            elif kind == 'sparse':
                cc = [(float(rng.uniform(12, 116)), float(rng.uniform(12, 116)), a, h, w) for cy, cx, a, h, w in centers]
            else:
                cc = centers
            f = self.pattern(rng, cc, kind) * t
            if kind == 'separated':
                f[:, :int(128 * j / len(active)) + 1] = 0
                f[:, int(128 * (j + 1) / len(active)) - 1:] = 0
            apeak = peak / (spec.get('ratio', 1) if j == spec.get('weak', -1) and len(active) > 1 else 1)
            x[k] = f * (apeak / max(float(f.max()), 1e-08))
        clean = np.einsum('mk,khw->mhw', self.matrix, x, optimize=True)
        gain = min(1.0, 240 / max(float(clean.max()), 1e-08))
        x *= gain
        sigma = spec.get('sigma', 0.0)
        factor = spec.get('factor', 1)
        if sigma:
            x = gaussian_filter(x, sigma=(0, sigma, sigma), mode='reflect', truncate=3).astype(np.float32)
        if factor > 1:
            x = x.reshape(13, 128 // factor, factor, 128 // factor, factor).mean((2, 4)).repeat(factor, 1).repeat(factor, 2)
        assert np.isfinite(x).all() and x.min() >= 0
        return (np.asarray(x, np.float32), origins)

    def spec_for(self, i, rng):
        z = i % 20
        kind = 'single' if z < 4 else 'partial' if z < 10 else 'colocated' if z < 12 else 'separated' if z < 15 else 'sparse' if z < 19 else 'blank'
        active = [] if kind == 'blank' else [i % 13] if kind == 'single' else sorted((int(v) for v in rng.choice(13, int(rng.integers(3, 6)), replace=False))) if kind == 'sparse' else self.pairs[(i // 20 + i) % len(self.pairs)] if i % 2 == 0 else sorted((int(v) for v in rng.choice(13, 2, replace=False)))
        return {'kind': kind, 'active': active, 'ratio': [1, 5, 20, 100][i % 4], 'weak': i // 4 % 2, 'sigma': 0.0 if i % 4 == 0 else float(rng.uniform(0.3, 1.5)), 'factor': 2 if i % 4 == 3 else 1}
