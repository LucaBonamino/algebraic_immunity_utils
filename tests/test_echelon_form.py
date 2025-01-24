import unittest
from algebraic_immunity_utils import Matrix as GF2Matrix

class TestEchelonForm(unittest.TestCase):

    def test_echelon_form_last_row_simple(self):
        m = GF2Matrix([[1,1], [1,0]])
        m, ops = m.echelon_form_last_row()
        self.assertEqual(m.to_list(), [[1,1],[0,1]])
        self.assertEqual(ops, [(1,0)])

    def test_echelon_form_last_row_no_change(self):
        m = GF2Matrix([[1,1], [0,1]])
        m, ops = m.echelon_form_last_row()
        self.assertEqual(m.to_list(), [[1,1],[0,1]])
        self.assertEqual(ops, [])

        m = GF2Matrix([[1,0], [0,0]])
        m, ops = m.echelon_form_last_row()
        self.assertEqual(m.to_list(), [[1,0],[0,0]])
        self.assertEqual(ops, [])