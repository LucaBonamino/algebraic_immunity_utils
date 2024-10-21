import algebraic_immunity_utils

print(algebraic_immunity_utils.sum_as_string(1,2))

print(algebraic_immunity_utils.echelon_form_last_row([[1, 0, 0], [1,1,0]]))

m = algebraic_immunity_utils.Matrix([[1,1], [1,0]])
print(m.echf_2())