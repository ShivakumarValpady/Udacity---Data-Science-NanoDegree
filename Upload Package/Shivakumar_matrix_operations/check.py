from matrix_operations import MatrixOperations
import numpy as np

mat_one = 3 * np.ones((3,1))
mat_two = 2 * np.ones((1,3))
mat_fun = MatrixOperations(mat_one, mat_two)


print(mat_fun.add_matrices())
print(mat_fun.subtract_matrices())
print(mat_fun.multiply_matrices())
print(mat_fun.transpose_of_matrix(mat_one))
print(mat_fun.scalar_multiplication(mat_one, 2))

