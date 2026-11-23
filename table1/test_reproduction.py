"""Check configurations, data assignments, and numerical definitions."""
import unittest
import numpy as np
from common import CONFIGURATIONS, ROOT, config, metrics, read_json
from evaluate import nnls

class ReproductionTests(unittest.TestCase):
    def test_assignments(self):
        manifest=read_json(ROOT/'dataset.json')
        self.assertEqual(len(manifest['train']),500)
        self.assertEqual(len(manifest['test']),50)
        self.assertEqual([len(p) for p in manifest['priors']],[324,340,272,280,492])
        paths=[r['file'] for r in manifest['train']]+[r['input'] for r in manifest['test']]
        self.assertEqual(len(paths),len(set(paths)))

    def test_matched_settings(self):
        rows=[config(n) for n in CONFIGURATIONS]
        for key in ['steps','epochs','A','B','response','scale','seed_G','seed_D','seed_data','seed_prior','initial_G_SHA256','initial_D_SHA256']:
            self.assertTrue(all(r[key]==rows[0][key] for r in rows))
        self.assertEqual(rows[0]['steps'],45000)
        a=config('full');b=config('reconstruction_image_prior')
        for key in a:
            if key not in ('name','lambda_denoising'):self.assertEqual(a[key],b[key])

    def test_identical_metrics(self):
        a=np.random.default_rng(7).uniform(1,1000,(5,16,16)).astype(np.float32)
        r=metrics(a,a)
        self.assertEqual(r['PSNR'],100.)
        self.assertAlmostEqual(r['SSIM'],1.)
        self.assertLess(r['SAM'],1e-5)

    def test_nonnegative_solution(self):
        matrix=np.asarray(config('full')['response'],np.float64)
        x=np.random.default_rng(9).uniform(0,2000,(5,3,3));y=np.einsum('ck,khw->chw',matrix,x)
        self.assertTrue(np.allclose(nnls(matrix,y),x,atol=1e-7,rtol=0))

if __name__=='__main__':unittest.main()
