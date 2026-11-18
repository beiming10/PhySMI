"""CPU smoke tests for the four generator interfaces."""
from pathlib import Path
import sys,unittest
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physmi.models import load_config,build_model,predict,prepare_input,ROOT

class ModelInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_published_examples(self):
        for name in ['static','live','cells','ecoli']:
            with self.subTest(model=name):
                c=load_config(name);model=build_model(c).eval()
                image=np.load(ROOT/'examples'/(name+'_input.npy'))
                expected=np.load(ROOT/'examples'/(name+'_output.npy'))
                panel=torch.ones(1,5) if c['known_panel'] else None
                with torch.inference_mode():_,out=predict(model,torch.from_numpy(image[None]),c,panel)
                np.testing.assert_allclose(out[0].numpy()*c['unit'],expected,rtol=1e-5,atol=1e-5)
    def test_known_panel_retains_five_outputs(self):
        c=load_config('static');model=build_model(c).eval()
        with torch.inference_mode():_,out=predict(model,torch.zeros(1,3,128,128),c,torch.tensor([[0,1,0,0,0.]]))
        self.assertEqual(tuple(out.shape),(1,5,128,128))
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(float(out[:,[0,2,3,4]].sum()),0.)
    def test_raw_selected_and_full_agree(self):
        for name in ['static','live']:
            c=load_config(name);full=np.arange(8*16*16,dtype=np.float32).reshape(8,16,16)
            np.testing.assert_array_equal(prepare_input(full,c,'raw'),prepare_input(full[c['A']],c,'raw'))
    def test_configuration_weight_pair(self):
        for name in ['static','live','cells','ecoli']:
            c=load_config(name)
            self.assertEqual(len(c['response']),c['reconstruction_channels'])
            self.assertEqual(len(c['response'][0]),len(c['components']))

if __name__=='__main__':unittest.main()
