import unittest

import numpy as np

from nnws import (
    get_symmetric_component,
    normalize_max_abs,
    training_rms_ratio,
)


class MetricsTest(unittest.TestCase):
    def test_symmetric_component_aligns_zero_lag(self):
        np.testing.assert_array_equal(
            get_symmetric_component([1, 2, 3, 4, 5]), [6, 6, 6]
        )

    def test_normalize_max_abs_is_safe_for_zero_rows(self):
        result = normalize_max_abs(np.array([[0.0, 0.0], [-2.0, 1.0]]))
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        np.testing.assert_allclose(result[1], [-1.0, 0.5])

    def test_sampling_rate_matches_equivalent_delta(self):
        signal = np.linspace(-1, 1, 41) ** 2 + 0.1
        with_rate = training_rms_ratio(
            signal, tau=1, tmin=3, tmax=5, sampling_rate=2
        )
        with_delta = training_rms_ratio(
            signal, tau=1, tmin=3, tmax=5, delta=0.5
        )
        self.assertAlmostEqual(with_rate, with_delta)


if __name__ == "__main__":
    unittest.main()
