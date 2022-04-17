import numpy as np
import copy



def determinant_laplacian2(theta):
    A = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i != j:
                A[i, j] = - theta[i, j]

    r = np.zeros(10)
    for m in range(10):
        A[m, m] = - np.sum(A[m, :])
        r[m] = theta[0, m]

    Z = 0
    for m in range(10):
        L = copy.deepcopy(A)
        L = np.delete(arr=L, obj=0, axis=0)
        L = np.delete(arr=L, obj=m, axis=1)
        Z += r[m] * np.linalg.det(L) *(-1)**(m)
    print(Z)


def determinant_laplacian3(theta):
    A = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if i != j:
                A[i, j] = - theta[i, j]

    r = np.zeros(10)
    for m in range(10):
        A[m, m] = - np.sum(A[m, :])
        r[m] = theta[0, m]

    A[0, :] = r
    Z = 0
    for m in range(10):
        L = copy.deepcopy(A)
        r = A[0, m]
        L = np.delete(arr=L, obj=0, axis=0)
        L = np.delete(arr=L, obj=m, axis=1)
        Z += r * np.linalg.det(L) *(-1)**(m)
        print(r, np.linalg.det(L))
    print(np.linalg.det(A))
    print(Z)

theta = np.abs(np.random.randn(20, 20) / 2)
# determinant_laplacian(theta)
print("-"*40)
determinant_laplacian2(theta)
print("-"*40)
determinant_laplacian3(theta)
