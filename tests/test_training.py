import unittest

import numpy as np

from nnws import TrainingConfig, train


class TrainingTest(unittest.TestCase):
    def test_sampling_rate_is_propagated_to_model(self):
        data = np.random.default_rng(4).normal(size=(4, 21))
        result = train(
            data,
            tau=0.5,
            tmin=1.5,
            tmax=2.5,
            sampling_rate=2,
            config=TrainingConfig(epochs=1, seed=3, device="cpu"),
        )
        self.assertEqual(result.model.sampling_rate, 2)
        self.assertEqual(result.model.delta, 0.5)
        self.assertEqual(result.model.lag, 5)
        self.assertEqual(result.model.input_dim, data.shape[1])


if __name__ == "__main__":
    unittest.main()
