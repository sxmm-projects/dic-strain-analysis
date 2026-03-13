# %%
import copy
import time
import numpy             as np
import scipy.linalg      as solver
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'L3'))
import L3_c1_linearSolve as linearS

eps = linearS.eps


# In Step 1: elementwise givens rotation to remove xb
# i = a, j = b
#                 ____________
#                /  2      2
# r = sign(xb)*\/ xa  +  xb
# 
# cos(theta) = c = xa/r
# sin(theta) = s = xb/r
def givensMatRotation(A, i, j, tol=eps):
    a, b = A[i, i], A[j, i]
    # zero element => no need to remove => skip
    if np.abs(b) < tol: 
        A[j, i] = 0
        return 

    # 2 rows (a and b) from column a on
    rows = A[i:j+1:j-i, i+1:]

    # case 1: s = 1, c = 0 (xa = 0)
    if np.abs(a) < tol:
        r                      = b
        rho                    = 1   # encoded
        rows[0, :], rows[1, :] = rows[1, :], -rows[0, :]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                r   = np.sqrt(a*a + b*b)
            # error: r -> inf as a and/or b -> inf
            except Warning as e:
                fac = np.abs(a) if np.abs(a) > np.abs(b) else np.abs(b)
                r   = fac*np.sqrt(((a/fac)*a + (b/fac)*b)/fac)

        if b < 0: r = -r
        s, c = b/r, a/r
        
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                # encoding
                if c < 0: # negative cos
                    if s < -c: rho = -s/2 # case 2
                    else:      rho = 2/c  # case 3
                else:     # zero or positive cos
                    if s < c:  rho = s/2  # case 2
                    else:      rho = 2/c  # case 3
            # error: rho -> inf as c -> 0 (case 1)
            except Warning as e: 
                r                      = b
                # case 1: s = 1, c = 0
                rho                    = 1   # encoded
                rows[0, :], rows[1, :] = rows[1, :], -rows[0, :]

        # case 2 or 3
        if rho != 1: rows[:, :] = np.array([[c, s], [-s, c]])@rows

    # xa replaced by r while xb, actually 0, stores rho for later use
    A[i, i], A[j, i] = r, rho

    return


# In Step 2: matrix-vector multiplication = Ga,b*b 
#            by decoding the stored rho to recompose Ga,b
# i = a, j = b

# 1. decoding rho to obtain cos(theta) and sin(theta)
def givensDecoding(A, i, j):
    rho = A[j, i] # code
    
    # case 0
    if rho == 0:
        s, c = None, None
    # case 1
    elif rho == 1:
        s, c = 1, 0
    # case 2
    elif np.abs(rho) < 1:
        if rho < 0:
            s = -2*rho
            c = -np.sqrt(1 - s*s)
        else:
            s = 2*rho
            c = np.sqrt(1 - s*s)
    # case 3
    else:
        c = 2/rho
        s = np.sqrt(1 - c*c)

    return s, c

def givensVecRotation(A, i, j, b):
    # 1. decoding rho to obtain cos(theta) and sin(theta)
    s, c = givensDecoding(A, i, j)
    if s is None: return # vector rotation skipped
    
    # 2. matrix-vector multiplication directly to 
    #    the rows a and b of the RHS vector b
    b[i], b[j] = c*b[i] + s*b[j], c*b[j] - s*b[i]

    return


# QR decomposition
# Step 1: composing R and encoding rho
def givensDecomposition(A, m, n, elRange, tol=eps):
    R = copy.deepcopy(A)

    if m > n: col = n
    else:     col = n - 1

    for i in range(col):
        for j in range(elRange[i, 0], elRange[i, 1]):
            givensMatRotation(R, i, j, tol)
    
    return R


# Step 2: serial matrix-vector multiplications = Gm,n*...*G1,2*b 
def givensRHStall(R, b, m, n, elRange):
    Qb = copy.deepcopy(b)

    if m > n: col = n     # tall matrix
    else:     col = n - 1 # square matrix

    for i in range(col):
        for j in range(elRange[i, 0], elRange[i, 1]):
            givensVecRotation(R, i, j, Qb)

    return Qb


def givensSolve(A, b):
    m, n    = np.shape(A)
    elRange = np.array([np.arange(1, n+1), np.full(n, m)], dtype=int).T
    if m < n:
        print("\nGivens rotation is not applicable for a fat system matrix\n")
        return None, None
    else:
        # 1. QR decomposition (composing R)
        R     = givensDecomposition(A, m, n, elRange, eps)
        # 2. serial matrix-vector multiplications to b: Qb = Gm,n*...*G1,2*b
        Qb    = givensRHStall(R, b, m, n, elRange)
        # 3. back substitution
        x_hat = linearS.backSub(R, Qb, None, None, R)

    return x_hat, R


# A_hat = Q*R = Gt1,2*Gt1,3*...*Gtn,m*R
def AhatGivens(Rrho):
    (m, n)                            = Rrho.shape
    R                                 = copy.deepcopy(Rrho)
    R[np.tril_indices_from(Rrho, -1)] = 0
    QtR                               = copy.deepcopy(R)

    if m > n: col = n       # tall matrix
    else:     col = n - 1   # square matrix

    # from column n-1 back to 0
    for i in range(col-1, -1, -1):
        # from row m-1 up to i+1
        for j in range(m-1, i, -1):
            s, c = givensDecoding(Rrho, i, j)
            if s is None: continue
            QtR[i, :], QtR[j, :] = c*QtR[i, :] - s*QtR[j, :], c*QtR[j, :] + s*QtR[i, :]

    return QtR, R


if __name__ == "__main__":
    # A = np.array([[3, 5], [2, 2], [0, 0], [5, 4]], dtype=float)
    # A = np.array([[3, 14], [2, 70], [0, 0], [6, 35]], dtype=float)
    # A = np.array([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]], dtype=float)
    A = np.array([[1, -5], [2, 0], [2, 1]], dtype=float)
    b = np.array([2, -11, -5], dtype=float)
    # b = np.ones(A.shape[0], dtype=float)
    # x = np.ones(A.shape[1], dtype=float)
    # b = A@x
    
    xhat, R  = givensSolve(A, b)
    print(f"\nxhat:\n{xhat}\nR:\n{R}\n")

    QRgivens, Rhat = AhatGivens(R)
    print(f"\nAhat:\n{QRgivens}\n")




# %%
