"""Shared model, sampling, and scoring functions."""
from pathlib import Path
import copy
import hashlib
import json
import random
import numpy as np
import tifffile
import torch
from skimage.metrics import structural_similarity
from network import FSR_Unmixing
from discriminator import StCDiscriminator

ROOT = Path(__file__).resolve().parent
CONFIGURATIONS = ('reconstruction_only', 'reconstruction_cross_denoising',
                  'reconstruction_image_prior', 'full', 'adversarial_prior_only')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def config(name):
    if name not in CONFIGURATIONS:
        raise ValueError('Unknown configuration')
    return read_json(ROOT / 'configs' / (name + '.json'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def state_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def build(c, device='cpu', initial=None):
    torch.manual_seed(c['seed_G'])
    matrix = np.asarray(c['response'], np.float32)
    a = torch.from_numpy(matrix.T.copy())
    g = FSR_Unmixing(in_channels=3, out_channels=8, num_endmembers=5,
                    signal_init=a[:4], background_init=a[4],
                    freeze_signal=True, freeze_background=True)
    torch.manual_seed(c['seed_D'])
    template = StCDiscriminator(in_channels=1)
    d = torch.nn.ModuleList([copy.deepcopy(template)])
    if initial is not None:
        state = torch.load(initial, map_location='cpu', weights_only=True)
        g.load_state_dict(state['G'], strict=True)
        d.load_state_dict(state['D'], strict=True)
    if state_digest(g) != c['initial_G_SHA256'] or state_digest(d[0]) != c['initial_D_SHA256']:
        raise RuntimeError('Initialization does not match the configuration')
    return g.to(device), d.to(device)

class Observations:
    def __init__(self, data, c, with_priors=True):
        self.root = Path(data); self.c = c
        self.manifest = read_json(ROOT / 'dataset.json')
        self.train = []
        for item in self.manifest['train']:
            self.train.append(tifffile.imread(self.root / item['file']))
        self.train = np.stack(self.train)
        if self.train.shape != (500, 8, 256, 256) or self.train.dtype != np.uint16:
            raise ValueError('Invalid observation stack')
        self.pools = []
        if with_priors:
            for pool in self.manifest['priors']:
                images = []
                for item in pool:
                    v = tifffile.imread(self.root / item['file'])
                    v = v[0] if v.ndim != 2 else v
                    images.append(v.astype(np.float32) / c['scale'])
                self.pools.append(images)
        self.orders = {}; self.prior_orders = {}

    def batch(self, epoch, batch):
        c = self.c
        if epoch not in self.orders:
            self.orders.clear()
            self.orders[epoch] = np.random.default_rng(np.random.SeedSequence([c['seed_data'], epoch])).permutation(4500)
        indices = self.orders[epoch][batch * 4:batch * 4 + 4]
        rng = random.Random(c['seed_data'] * 100000000 + epoch * 10000 + batch)
        images = []
        for index in indices:
            image, patch = divmod(int(index), 9)
            y, x = (patch // 3) * 64, (patch % 3) * 64
            v = self.train[image, :, y:y+128, x:x+128].astype(np.float32) / c['scale']
            turns, vflip, hflip = rng.randint(0, 3), rng.randint(0, 1), rng.randint(0, 1)
            v = np.rot90(v, turns, axes=(1, 2))
            if vflip: v = v[:, :, ::-1]
            if hflip: v = v[:, ::-1, :]
            images.append(np.ascontiguousarray(v))
        return torch.from_numpy(np.stack(images))

    def prior_batch(self, index, branch, component):
        c = self.c; pool = self.pools[component]; n = len(pool)
        images = []
        for j in range(4):
            position = ((index - 1) * 2 + branch) * 4 + j
            cycle, ix = divmod(position, n * 100)
            key = (component, cycle, 0)
            if key not in self.prior_orders:
                if len(self.prior_orders) > 24: self.prior_orders.clear()
                self.prior_orders[key] = np.random.default_rng(np.random.SeedSequence([c['seed_prior'], component, cycle, 0])).permutation(n * 100)
            v = pool[int(self.prior_orders[key][ix]) % n]
            rng = random.Random(c['seed_prior'] * 1000000000 + index * 10000 + branch * 1000 + component * 100 + j)
            y, x = rng.randint(0, v.shape[0]-128), rng.randint(0, v.shape[1]-128)
            v = v[y:y+128, x:x+128]
            if rng.randint(0, 1): v = v[:, ::-1]
            if rng.randint(0, 1): v = v[::-1, :]
            images.append(np.ascontiguousarray(v))
        return torch.from_numpy(np.stack(images)[:, None])

def metrics(prediction, reference):
    p = np.asarray(prediction, np.float64); r = np.asarray(reference, np.float64)
    if p.shape != r.shape or p.shape[0] != 5 or not np.isfinite(p).all():
        raise ValueError('Expected finite matching five-component arrays')
    mse = ((p-r)**2).mean((1, 2))
    psnr = np.where(mse == 0, 100., 10 * np.log10(5000.**2 / np.maximum(mse, 1e-30)))
    ssim = [float(structural_similarity(p[k].astype(np.float32), r[k].astype(np.float32), data_range=5000, win_size=7)) for k in range(5)]
    den = np.linalg.norm(p, axis=0) * np.linalg.norm(r, axis=0)
    den[den == 0] = 1e-10
    sam = np.arccos(np.clip((p*r).sum(0) / den, -1, 1)) * 180 / np.pi
    return dict(PSNR=float(psnr.mean()), SSIM=float(np.mean(ssim)), SAM=float(sam.mean()),
                component_PSNR=psnr.tolist(), component_SSIM=ssim)
