# %%
import numpy as np
import copy
import sys

eps = sys.float_info.epsilon


# back substitution
# division by diagonals
# 1. None: by one (no division)
# 2. A: by the diagonals of A
# 3. diag: by the diagonals of the diag vector
def backSub(A, b, pA, pb, diag=None):
    n = np.shape(A)[1]
    x = np.zeros(n, dtype=float)
    if pA is None:
        pA = np.arange(n)
        if pb is None:
            pb = pA
    elif pb is None:
        pb = np.arange(n)
    
    if diag is A:          diag = A[pA, np.arange(n)]
    elif diag is not None: diag = diag[pA] # diag vector
    # otherwise diag is None

    #                  n 
    # x_i = ( b_i −   sum (A_ij*x_j) ) / A_ii
    #                j=i+1
    for i in range(n-1,-1,-1):
        x[i] =  b[pb[i]]
        for j in range(i+1, n):
            x[i] -= A[pA[i], j]*x[j]
        if diag is not None:
            x[i] /= diag[i]
    return x

# forward substitution
# division by diagonals
# 1. None: by one (no division)
# 2. A: by the diagonals of A
# 3. diag: by the diagonals of the diag vector
def forwSub(A, b, pA, pb, diag=None):
    n = np.shape(A)[1]
    x = np.zeros(n, dtype=float)
    if pA is None:
        pA = np.arange(n)
        if pb is None:
            pb = pA
    elif pb is None:
        pb = np.arange(n)

    if diag is A:          diag = A[pA, np.arange(n)]
    elif diag is not None: diag = diag[pA] # diag vector
    # otherwise diag is None

    for i in range(n):
        x[i] = b[pb[i]]
        for j in range(i):
            x[i] -= A[pA[i], j]*x[j]
        if diag is not None:
            x[i] /= diag[i]
    return x


# pivoting
def pivoting(Aj, p, j, n):
    pInd = j
    big  = np.abs(Aj[p[j]])
    for k in range(j+1, n):
        temp = np.abs(Aj[p[k]])
        if temp > big:
            pInd = k
            big  = temp
    if pInd != j:
        k       = p[j]
        p[j]    = p[pInd]
        p[pInd] = k
    return

def gaussElimination(A, b, pivot):
    if np.isscalar(A): return A, None 
    n    = A.shape[0]
    augA = np.append(A, b.reshape(n, 1), axis=1)
    p    = np.arange(n)
    for j in range(n-1):
        Lj  = augA[:, j]
        if pivot is True: 
            pivoting(Lj, p, j, n)
        piv = augA[p[j], j]   # pivot element
        j1  = j + 1
        Up  = augA[p[j], j1:] # pivot row
        # elementary row operations
        # (A|b)_i = (A|b)_i - A_ij/A_jj*(A|b)_j
        for i in p[j1:]:
            augA[i, j1:] -= (Lj[i]/piv)*Up
            Lj[i]        =  0

    # back substitution
    mat   = augA[:, :n]
    x_hat = backSub(mat, augA[:, n], p, p, mat)

    return x_hat

# 1. LU decomposition
def LUdecomposition(A, pivot):
    if np.isscalar(A): return A, None

    n  = A.shape[0]
    lu = copy.deepcopy(A)
    p  = np.arange(n)
    for j in range(n-1):
        Lj  = lu[:, j]
        if pivot is True: 
            pivoting(Lj, p, j, n)
        piv = lu[p[j], j]   # pivot element
        j1  = j + 1
        Up  = lu[p[j], j1:] # pivot row
        for i in p[j1:]:
            Lj[i]      /= piv
            lu[i, j1:] -= Lj[i]*Up

    return lu, p

# 2. and 3. forward and back substitutions
def LUsolve(lu, p, b):
    if np.isscalar(lu): return b/lu

    # 2. forward substitution
    # all-one diagonals
    y_hat = forwSub(lu, b, p, p, None)
    
    # 3. back substitution
    # y_hat inherently pivoted (no more pivot needed)
    x_hat = backSub(lu, y_hat, p, None, lu)

    return x_hat

# LU solver
def LUsolver(A, b, pivot):
    # 1. LU decomposition
    lu, p = LUdecomposition(A, pivot)
    # 2. and 3. forward and back substitutions
    x_hat = LUsolve(lu, p, b)

    return x_hat, lu, p

# A_hat = L*U
def AhatLU(lu, pLU):
    n    = lu.shape[0]
    A    = copy.deepcopy(lu)
    L    = np.zeros((n, n), dtype=float)
    Ahat = np.zeros((n, n), dtype=float)

    for i in range(n):
        L[i, :] = A[pLU[i], :]

    U = copy.deepcopy(L)

    for i in range(n):
        L[i, i]    = 1
        L[i, i+1:] = U[i, :i] = 0
    
    A = L@U
    for i in range(L.shape[0]):
        Ahat[pLU[i], :] = A[i, :]

    return Ahat, L, U

def oneMat(n):
    A    = np.zeros((n, n), dtype=float)
    last = n - 1
    for i in range(n):
        for j in range(n):
            if i > j:
                A[i][j] = -1
            elif (i == j) or (j == last):
                A[i][j] = 1
    return A

def vandermondemat(m, n):
    step = 1.0/(n - 1)
    xj   = 0
    A    = np.ones((m, n), dtype=float)
    x    = np.arange(1, n+1, dtype=float)
    for j in range(n-1):
        xij = xj
        for i in range(1, m):
            A[i, j] =  xij
            xij     *= xj
        xj += step

    return A, x

if __name__ == "__main__":
    n = 10
    # A = oneMat(n)
    # x = np.ones(n)
    # b = A@x
    
    A    = np.array([[1, 1, 1, 1], [-3, 2, 0, 0], [2, -5, 3, 0], [0, 3, -7, 4]], dtype=float)
    b    = np.array([1.2, 0, 0, 0], dtype=float)
    # A    = np.array([[-1, 2, 0, 0], [0, -2, 3, 0], [0, 0, -3, 4], [1, 1, 1, 1]], dtype=float)
    # b    = np.array([0, 0, 0, 3.4], dtype=float)

    xhatLU, lu, pLU = LUsolver(A, b, False)
    print(f"\nxhat:\n{xhatLU}\nLU:\n{lu}\n")

    LUAhat, L, U = AhatLU(lu, pLU)
    print(f"\nAhat:\n{LUAhat}\n")

# %%
