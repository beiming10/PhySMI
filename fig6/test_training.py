"""Check source assignments, schedules, and observation interfaces."""
from pathlib import Path
import unittest
import math
import numpy as np
from common import ROOT,configuration,learning_rate,read,scores
from simulation import EcoliSimulator

class TrainingTests(unittest.TestCase):
    def test_source_assignments(self):
        rows=read(ROOT/'library.json')['sources']
        self.assertEqual(len(rows),1108)
        for model in ['cells','ecoli']:
            clusters={}
            for row in rows:
                value=clusters.setdefault(row['cluster'],row[model+'_split'])
                self.assertEqual(value,row[model+'_split'])
        self.assertEqual(sum(r['cells_split']=='train' for r in rows),896)
        self.assertEqual(sum(r['cells_split']=='holdout' for r in rows),212)

    def test_schedules(self):
        with np.load(ROOT/'orders.npz',allow_pickle=False) as orders:
            for model,n in [('cells',2048),('ecoli',1024)]:
                c=configuration(model);self.assertEqual(orders[model].shape,(c['steps'],2))
                self.assertGreaterEqual(int(orders[model].min()),0)
                self.assertLess(int(orders[model].max()),n)
                self.assertAlmostEqual(learning_rate(c,1),2e-4)
                self.assertAlmostEqual(learning_rate(c,20000),4e-5)
        self.assertAlmostEqual(learning_rate(configuration('cells'),20001),4e-5)
        self.assertAlmostEqual(learning_rate(configuration('cells'),40000),8e-6)

    def test_bacterial_inputs_are_disjoint(self):
        c=configuration('ecoli');groups=c['A_groups'];flat=sum(groups,[])
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(len(groups),5)
        self.assertEqual(c['input_channels'],5)
        self.assertEqual(len(c['components']),13)

    def test_independent_cell_inputs(self):
        c=configuration('cells')
        self.assertEqual(c['input_channels'],3)
        self.assertEqual(c['reconstruction_channels'],28)
        self.assertEqual(c['B'],[10,15,23])
        self.assertEqual(sum(len(x) for x in c['A_groups']),22)

    def test_missing_component_scores(self):
        r=np.ones((5,8,8));r[0]=0;p=r.copy();p[0]=.1
        values=scores(p,r)
        self.assertIsNone(values[0]['amplitude_slope'])
        self.assertAlmostEqual(values[0]['output_RMS'],.1)
        self.assertAlmostEqual(values[1]['amplitude_slope'],1.)

if __name__=='__main__':unittest.main()
