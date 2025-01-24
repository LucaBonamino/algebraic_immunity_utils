import unittest
from algebraic_immunity_utils import Matrix as GF2Matrix

class TestBasicMethods(unittest.TestCase):

    def test_to_list(self):
        m = GF2Matrix([[1,1], [1,0]])
        self.assertEqual(m.to_list(), [[1,1], [1,0]])

    def test_add_rows(self):
        m = GF2Matrix([[1,1], [1,0]])
        m.append_row([1,0])

        m_res = GF2Matrix([[1,1], [1,0], [1,0])
        self.assertEqual(m, m_res)