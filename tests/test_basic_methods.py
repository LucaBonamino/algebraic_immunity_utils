import unittest
from algebraic_immunity_utils import Matrix as GF2Matrix

class TestBasicMethods(unittest.TestCase):

    def test_to_list(self):
        m = GF2Matrix([[1,1], [1,0]])
        self.assertEqual(m.to_list(), [[1,1], [1,0]])

    def test_add_rows(self):
        m = GF2Matrix([[1,1], [1,0]])
        m.append_row([1,0])

        m_res = GF2Matrix([[1,1], [1,0], [1,0]])
        self.assertEqual(m, m_res)

    def test_compute_next(self):
        m = GF2Matrix([[1, 1], [1, 0]])
        support_slice = ['00', '01', '10']
        monom_slice = ['00', '01', '10']
        idx = 2
        operations = []
        n = m.compute_next(monom_slice, support_slice, idx, operations)
        self.assertEqual(n.to_list(), [[1, 1, 0], [1, 0, 0], [1, 0, 1]])

        m = GF2Matrix([[1,1], [0,1]])
        support_slice = ['000', '010', '100']
        monom_slice = ['000', '001', '010']
        idx = 2
        operations = [(0, 1)]
        n_m = m.compute_next(monom_slice, support_slice, idx, operations)
        self.assertEqual(n_m.to_list(), [[1, 1, 1], [0, 1, 1], [1, 0, 0]])
        self.assertEqual(m.to_list(), [[1,1], [0,1]])

    def construct_and_add_column(self):
        m = GF2Matrix([[1, 1], [1, 0]])
        support_slice = ['00', '01', '10']
        monom_slice = ['00', '01', '10']
        idx = 2
        operations = []
        n = m.compute_next(monom_slice, support_slice, idx, operations)
        self.assertEqual(n.to_list(), [[1, 1, 0], [1, 0, 0], [1, 0, 1]])