import numpy as np
from numpy.linalg import norm
np.set_printoptions(precision=2, suppress=True)

A = np.array([[1, 2, 0],[0,0,5], [3,-4,2],[1,6,5], [0,1,0]])
U, s, W = np.linalg.svd(A)
V = W.T
D = np.zeros_like(A, dtype=float)
D[np.diag_indices(min(A.shape))] = s

print(f'Матрица A:\n{A}')
print(f'Матрица D:\n{D}')
print(f'Матрица U:\n{U}')
print(f'Матрица W (Vт):\n{V}')
print(f'Евклидова норма матрицы А:\n{norm(A, ord=2)}')
print(f'Норма Фробениуса матрицы А:\n{norm(A, ord=None)}')