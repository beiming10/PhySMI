"""CPU interface, preparation and inference tests for the four public models."""
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from physmi.models import load_config,build_model,predict,prepare_input,ROOT
from physmi.inference_io import infer_array,read_input,write_prediction,starts


class ModelInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_normalized_input_examples(self):
        for name in ['static','live','cells','ecoli']:
            with self.subTest(model=name):
                c=load_config(name);model=build_model(c).eval()
                image=read_input(ROOT/'examples'/(name+'_input.tif'))
                self.assertEqual(image.shape[0],c['input_channels'])
                panel=[1]*5 if c['known_panel'] else None
                value=infer_array(model,image,c,panel=panel,tile_size=0)
                with torch.inference_mode():
                    _,direct=predict(model,torch.from_numpy(image[None]),c,
                                     torch.ones(1,5) if c['known_panel'] else None)
                np.testing.assert_allclose(value[0],direct[0].numpy()*c['unit'],rtol=0,atol=0)
                self.assertEqual(value.shape[1],len(c['components']))
                self.assertTrue(np.isfinite(value).all())

    def test_static_retains_five_unconditioned_outputs(self):
        c=load_config('static');model=build_model(c).eval()
        self.assertFalse(c['known_panel'])
        with torch.inference_mode():
            _,out=predict(model,torch.zeros(1,3,128,128),c)
        self.assertEqual(tuple(out.shape),(1,5,128,128))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(any(k.startswith('readout.context.') for k in model.state_dict()))

    def test_raw_selected_and_full_agree(self):
        for name in ['static','live']:
            c=load_config(name);full=np.arange(8*16*16,dtype=np.float32).reshape(8,16,16)
            np.testing.assert_array_equal(prepare_input(full,c,'raw'),
                                          prepare_input(full[c['A']],c,'raw'))

    def test_integrated_inputs_not_integrated_twice(self):
        for name in ['cells','ecoli']:
            c=load_config(name)
            full=np.arange(c['reconstruction_channels']*8*9,dtype=np.float32).reshape(-1,8,9)
            selected=np.stack([full[g].sum(0) for g in c['A_groups']])
            np.testing.assert_array_equal(prepare_input(full,c,'raw'),
                                          prepare_input(selected,c,'raw'))

    def test_configuration_weight_pair(self):
        for name in ['static','live','cells','ecoli']:
            c=load_config(name)
            self.assertEqual(len(c['response']),c['reconstruction_channels'])
            self.assertEqual(len(c['response'][0]),len(c['components']))

    def test_invalid_input_is_rejected(self):
        c=load_config('static')
        for image in [np.zeros((4,16,16)),np.full((3,8,8),np.nan),np.zeros((3,0,8))]:
            with self.assertRaises(ValueError):prepare_input(image,c,'normalized')

    def test_tile_starts_cover_edges(self):
        self.assertEqual(starts(256,128,64),[0,64,128])
        self.assertEqual(starts(257,128,64),[0,64,128,129])

    def test_positive_hann_overlap_preserves_an_identity_map(self):
        image=np.random.default_rng(42).normal(size=(3,151,187)).astype(np.float32)
        c=dict(name='static',input_channels=3,unit=1.,components=['one','two','three'],known_panel=False)
        def identity(model,value,config,panel=None):return value,value
        with patch('physmi.inference_io.predict',side_effect=identity):
            result=infer_array(None,image,c,tile_size=64,tile_overlap=32)
        np.testing.assert_allclose(result[0],image,rtol=0,atol=1e-6)

    def test_npy_and_tiff_output(self):
        c=load_config('static')
        expected=np.arange(5*8*9,dtype=np.float32).reshape(1,5,8,9)
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ['npy','tif']:
                path=Path(directory)/('output_'+suffix+'.'+suffix)
                write_prediction(path,expected,c)
                np.testing.assert_array_equal(read_input(path),expected)
                info=json.loads(path.with_suffix('.json').read_text())
                self.assertEqual(info['components'],c['components'])
                with self.assertRaises(FileExistsError):write_prediction(path,expected,c)

    def test_live_panel_is_required(self):
        c=load_config('live')
        with self.assertRaises(ValueError):
            infer_array(None,np.zeros((3,8,8),np.float32),c)

    def test_testdata_channels_and_manifest(self):
        path=ROOT.parent/'TestDATA/examples.json'
        if not path.exists():self.skipTest('Full repository TestDATA is not present')
        examples=json.loads(path.read_text(encoding='utf-8'))['examples']
        self.assertEqual(len({e['id'] for e in examples}),len(examples))
        for e in examples:
            self.assertEqual(Path(e['file']).suffix.lower(),'.tif')
            image=read_input(path.parent/e['file'])
            expected=5 if e['model']=='ecoli' else 3
            self.assertEqual(image.ndim,3)
            self.assertEqual(image.shape[0],expected)
            self.assertTrue(np.isfinite(image).all())


if __name__=='__main__':unittest.main()
