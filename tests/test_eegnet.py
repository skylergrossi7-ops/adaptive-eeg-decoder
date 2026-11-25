"""Signal units, train/eval state, architecture constraints and CPU reproducibility."""

import unittest

import numpy as np
import torch

from eeg_bci.eegnet import EEGNet
from eeg_bci.train_eegnet import as_microvolts, fit_eegnet, predict_proba


class EEGNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.volts = np.random.default_rng(40).normal(scale=1e-6, size=(6, 22, 750))
        cls.y = np.array([0, 1, 0, 1, 0, 1])
        cls.settings = dict(seed=12, epochs=1, batch_size=3, learning_rate=0.001)

    def test_units_and_axes(self):
        self.assertEqual(as_microvolts(np.array([2e-6])).item(), 2.0)
        with self.assertRaisesRegex(ValueError, "750"):
            EEGNet()(torch.zeros(2, 750, 22))

    def test_rng_reproducibility_projection_and_frozen_evaluation(self):
        model = fit_eegnet(self.volts, self.y, **self.settings)
        buffers = {key: value.clone() for key, value in model.named_buffers()}
        first = predict_proba(model, self.volts)
        np.testing.assert_allclose(first.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(predict_proba(model, self.volts[:1]), first[:1], atol=1e-6)
        for key, value in model.named_buffers():
            self.assertTrue(torch.equal(value, buffers[key]))
        other = fit_eegnet(self.volts, self.y, **self.settings)
        np.testing.assert_array_equal(first, predict_proba(other, self.volts))
        for weight, cap in ((model.spatial.weight, 1.0), (model.classifier.weight, 0.25)):
            self.assertLessEqual(weight.detach().flatten(1).norm(dim=1).max().item(), cap + 1e-6)

    def test_missing_class_and_nonfinite_signal_fail(self):
        with self.assertRaisesRegex(ValueError, "both binary"):
            fit_eegnet(self.volts, np.zeros(6), **self.settings)
        broken = self.volts.copy()
        broken[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            fit_eegnet(broken, self.y, **self.settings)
