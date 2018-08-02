import numpy as np

print('Matrix multiplication using numpy')
# matrix_a = np.matrix(np.ones(shape=[3,3]))
matrix_a = np.matrix([[2,1,5],[1,-1,2],[2,1,0]])
print('matrix_a')
print(matrix_a)

matrix_b = np.matrix([[1,2],[3,4],[5,6]])
print('matrix_b')
print(matrix_b)

print()
multiplication = matrix_a * matrix_b
print('matrix_a * matrix_b')
print(multiplication)
print(multiplication.shape)

print()
print('matrix_c')
matrix_c = np.matrix([[2,1],[3,6],[7,5]])
print(matrix_c)
print('matrix_b .* matrix_c')
dot_mult = np.inner(matrix_b, matrix_c)
print(dot_mult)
print('np.multiply(matrix_b, matrix_c)')
print(np.multiply(matrix_b, matrix_c))

print()
array_b = np.array([[1,2],[3,4],[5,6]])
print(array_b)

print()
array_c = np.array([[2,1],[3,6],[7,5]])
print(array_c)
print(array_b*array_c)