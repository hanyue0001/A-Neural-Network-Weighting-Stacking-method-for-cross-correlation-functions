import unittest

import numpy as np
import torch

from nnws import StackingNet, weighted_stack


class ModelTest(unittest.TestCase):
    def test_forward_is_normalized_weighted_mean(self):
        model = StackingNet(lag=2, delta=1, hidden_dims=())
        with torch.no_grad():
            linear = model.net[0]
            linear.weight.zero_()
            linear.bias.zero_()  # sigmoid -> identical 0.5 weights
        data = torch.tensor(
            [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32
        )
        stack, weights = model(data)
        self.assertEqual(stack.shape, (1, 5))
        self.assertEqual(weights.shape, (2, 1))
        torch.testing.assert_close(stack, data.mean(dim=0, keepdim=True))

    def test_forward_accepts_single_leading_dimension(self):
        model = StackingNet(lag=2, delta=1, hidden_dims=())
        stack, weights = model(torch.ones(1, 3, 5))
        self.assertEqual(stack.shape, (1, 5))
        self.assertEqual(weights.shape, (3, 1))

    def test_sampling_rate_controls_input_dimension(self):
        model = StackingNet(lag=2, sampling_rate=2, hidden_dims=())
        self.assertEqual(model.input_dim, 9)
        self.assertEqual(model.delta, 0.5)
        self.assertEqual(model.sampling_rate, 2)
        stack, _ = model(torch.ones(3, 9))
        self.assertEqual(stack.shape, (1, 9))

    def test_conflicting_sampling_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            StackingNet(lag=2, delta=1, sampling_rate=2)

    def test_forward_rejects_ambiguous_or_wrong_shapes(self):
        model = StackingNet(lag=2, delta=1, hidden_dims=())
        for shape in ((2, 3, 5), (5,), (3, 7)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.ones(shape))

    def test_numpy_weighted_stack_uses_weight_sum(self):
        data = np.array([[1.0, 3.0], [5.0, 7.0]])
        np.testing.assert_allclose(weighted_stack(data, [1.0, 3.0]), [4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
