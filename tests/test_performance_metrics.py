import unittest
import numpy as np
from speed.metrics.performance_metrics import *

class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        self.annotations = [np.array([0, 0, 0]), np.array([0, 0]), np.array([1, 1, 1, 1]), np.array([1, 1, 0, 0, 1])]
        self.perfect_predictions = [
            {'probs': np.array([0, 0, 0]), 'mask': np.array([0, 0, 0])},
            {'probs': np.array([0, 0]), 'mask': np.array([0, 0])},
            {'probs': np.array([1, 1, 1, 1]), 'mask': np.array([1, 1, 1, 1])},
            {'probs': np.array([1, 1, 0, 0, 1]), 'mask': np.array([1, 1, 0, 0, 1])}
        ]
        self.random_predictions = [
            {'probs': np.array([0.5, 0.2, 0.8]), 'mask': np.array([1, 0, 1])},
            {'probs': np.array([0.7, 0.3]), 'mask': np.array([0, 1])},
            {'probs': np.array([0.1, 0.9, 0.2, 0.8]), 'mask': np.array([0, 1, 0, 1])},
            {'probs': np.array([0.9, 0.1, 0.3, 0.7, 0.6]), 'mask': np.array([1, 0, 0, 1, 1])}
        ]

    def test_count_babies_with_seizures(self):
        self.assertEqual(count_babies_with_seizures(self.annotations), 2)

    def test_auc_cc(self):
        self.assertEqual(auc_cc(self.annotations, self.perfect_predictions), 1.0)
        self.assertEqual(auc_cc(self.annotations, self.random_predictions), 0.51)

    def test_sensitivity(self):
        self.assertEqual(sensitivity(self.annotations, self.perfect_predictions), 1.0)
        self.assertEqual(sensitivity(self.annotations, self.random_predictions), 0.571)

    def test_specificity(self):
        self.assertEqual(specificity(self.annotations, self.perfect_predictions), 1.0)
        self.assertEqual(specificity(self.annotations, self.random_predictions), 0.429)

    def test_ppv(self):
        self.assertEqual(ppv(self.annotations, self.perfect_predictions), 1.0)
        self.assertEqual(ppv(self.annotations, self.random_predictions), 0.5)

    def test_npv(self):
        self.assertEqual(npv(self.annotations, self.perfect_predictions), 1.0)
        self.assertEqual(npv(self.annotations, self.random_predictions), 0.5)


if __name__ == '__main__':
    unittest.main()
