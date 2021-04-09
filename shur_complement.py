import numpy as np
import time
from copy import deepcopy
from scipy.spatial.distance import squareform

def shur_complement():
    n = 1024
    M = np.abs(np.random.randn(n, n))
    l1 = n // 2
    l2 = n - l1
    A = M[0:l1, 0:l1]
    B = M[0:l1, l1:]
    C = M[l1:, 0:l1]
    D = M[l1:, l1:]
    inv_D = np.linalg.inv(D)
    top_left_one = np.eye(l1)
    top_right_zero = np.zeros((l1, l2))
    bottom_left_zero = np.zeros((l2, l1))
    bottom_right_one = np.eye(l2)
    B_times_inv_D = np.matmul(B, inv_D)
    one_part = np.block([[top_left_one, B_times_inv_D],
                         [bottom_left_zero, bottom_right_one]])

    two_part = np.block([[A - np.matmul(B_times_inv_D, C), top_right_zero],
                         [bottom_left_zero, D]])

    three_part = np.block([[top_left_one, top_right_zero],
                           [np.matmul(inv_D, C), bottom_right_one]])
    print(np.sum(np.matmul(np.matmul(one_part, two_part), three_part) - M))
    # print(M)



# compute determint
def schur_complement_determinant(M, n):
    if n == 1:
        return M[0, 0]
    l1 = n // 2
    l2 = n - l1
    A = M[0:l1, 0:l1]
    B = M[0:l1, l1:]
    C = M[l1:, 0:l1]
    D = M[l1:, l1:]
    inv_A = np.linalg.inv(A)
    one_part = D - np.matmul(np.matmul(C, inv_A), B)
    return schur_complement_determinant(A, l1) * schur_complement_determinant(one_part, l2)


# use1 = 0
# use2 = 0
# for i in range(10):
#     print(i)
#     n = 2048
#     M = np.abs(np.random.randn(n, n) / 8)
#     st = time.time()
#     np.linalg.det(M)
#     use1 = time.time() - st
#     st = time.time()
#     shur_complement_determinant(M, n)
#     use2 = time.time() - st
# print(use1, use2)


# det_+(L)=det(L_{[V2, V2]}) det_+(SC(L,V1))
def DEI_APPROX ():
    n = 100
    # A need to be a square matrix
    A = squareform(np.random.random(n * (n - 1) // 2))
    np.einsum('ii->i', A)[:] = np.random.random(n)
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i, j] = - A[i, j]

    for i in range(n):
        L[i, i] = - np.sum(L[i, :])

    def det_plus(M, size):
        L_plus = deepcopy(M)
        L_plus = np.delete(arr=L_plus, obj=size - 1, axis=0)
        L_plus = np.delete(arr=L_plus, obj=size - 1, axis=1)
        return np.linalg.det(L_plus)
    l1 = n // 2
    L_v1_v1 = L[0:l1, 0:l1]
    L_V1_V2 = L[0:l1, l1:]
    L_V2_V1 = L[l1:, 0:l1]
    L_V2_V2 = L[l1:, l1:]
    inv_L_V2_V2 = np.linalg.inv(L_V2_V2)
    shur_complement_L_V1 = L_v1_v1 - np.matmul(np.matmul(L_V1_V2, inv_L_V2_V2), L_V2_V1)
    print(det_plus(L, n))
    print(np.linalg.det(L_V2_V2), det_plus(shur_complement_L_V1, l1),
          np.linalg.det(L_V2_V2) * det_plus(shur_complement_L_V1, l1))


shur_complement_determinant2()