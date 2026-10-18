import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from survival_agent import checkpoint_io
from survival_agent.muzero import Episode, MuZeroModel, mcts
from examples.inspect_model import main

torch.set_num_threads(1)

class ModelTests(unittest.TestCase):
    def test_finite_forward_without_environment(self):
        main()

    def test_search_visit_accounting(self):
        model=MuZeroModel(seed=1).eval()
        result=mcts(model,torch.zeros(1,4,5,5,5),simulations=2)
        self.assertEqual(int(result['visits'].sum()),2)
        self.assertAlmostEqual(float(result['policy'].sum()),1.)

    def test_bounded_structured_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.safe'
            checkpoint_io.save({'x':torch.arange(3),'array':np.zeros(2,np.float32)},p)
            result=checkpoint_io.load(p)
            torch.testing.assert_close(result['x'],torch.arange(3))
            p.write_bytes(b'not a structured checkpoint')
            with self.assertRaises(ValueError):checkpoint_io.load(p)

    def test_malformed_episode_and_oversize_architecture(self):
        episode=Episode(np.zeros((2,3),np.float32),np.zeros(1,np.int64),np.zeros(1,np.float32),np.ones((1,5),np.float32)/5,np.zeros(2,np.float32),np.ones((1,1),bool),np.zeros(1,bool))
        with self.assertRaises(ValueError):episode.validate()
        with self.assertRaises(ValueError):MuZeroModel(feature_width=10**9)

if __name__=='__main__':unittest.main()
