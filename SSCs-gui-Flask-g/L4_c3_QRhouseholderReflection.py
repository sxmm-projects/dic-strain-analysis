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


# initial wi: column vector y stored in the form of a matrix with 1 column
def reflectionPlaneUnitNormalVec(wi, diag):
    norm     =  (wi.T@wi).item()      # inner product: yt.y (.item() to make it a scalar)
    if norm == 0: return 0            # zero vector [0, 0, ..., 0] => 0 diagonal
    # nonzero vector
    diag     =  np.sqrt(norm)         # 2-norm of vector y: ||y||
    if wi[0, 0] >= 0: diag = -diag    # diag = -sign(y1)*||y||
    # wt.w = ||y||^2 + ||y||^2 + 2*y1*sign(y1)*||y|| = 2*yt.y + 2*|y1|*||y||
    norm     += (diag - 2*wi[0, 0])*diag
    # w = y + sign(y1)*||y||*e
    wi[0, 0] -= diag
    # wi/||wi|| stored in the form of a matrix with 1 column
    wi       /= np.sqrt(norm)
    return diag                       # -sign(y1)*||y||


# composing R through QR decomposition
def householderDecomposition(A, m, n):
    Rw          = copy.deepcopy(A)
    Rdiag       = np.zeros(n, dtype=float) # storing the diagonals of R
    singularity = False

    if m > n: col = n                      # tall A
    else:     col = n - 1                  # square A

    for i in range(col):
        # wi
        wi       = Rw[i:, i:i+1]           # column vector y stored in the form of a matrix with 1 column
        # computing wi/||wi|| + storing it in the lower left part of Rw; storing the diagonal of R in Rdiag
        Rdiag[i] = reflectionPlaneUnitNormalVec(wi, Rdiag[i]) 
        # singular matrix (zero column vector)
        if not Rdiag[i]: singularity = True
        # R = QA = A - 2*w*wt*A/(wt.w)
        Ai = Rw[i:, i+1:]
        if Ai.size > 0: Ai -= wi@(2*(wi.T@Ai))
    if m == n: Rdiag[col] = Rw[col, col]   # square A

    return Rw, Rdiag, singularity


# serial matrix-matrix multiplications Q = Qw1*...*Qwn
# Rw = mxn matrix
def Qmatrix(Rw, m, n):
    # Q = I - 2*w*wt/||w||^2
    Q = np.identity(m) - Rw[:, 0:1]@(2*Rw[:, 0:1].T)

    #          /       w*wt \         /   w  \   /   wt \
    # A*Q = A*|I - 2*------- | = A - |A*----- |*|2*----- |
    #          \     ||w||^2/         \ ||w||/   \ ||w||/
    for i in range(1, n):
        j        =  i + 1
        Q[:, i:] -= (Q[:, i:]@Rw[i:, i:j])@(2*Rw[i:, i:j].T)

    return Q

# serial matrix-matrix multiplications Qy = (Qw1*...*Qwm)*y
def householderRHSfat(Rw, b, m, n):
    return Qmatrix(Rw, n, m)[:, :m]@b

# serial matrix-vector multiplications Qtb = (Qwn*(...*(Qw1*b)))
def householderRHStall(Rw, b, m, n):
    Qb = copy.deepcopy(b)

    if m > n: col = n     # tall matrix
    else:     col = n - 1 # square matrix
    #              w*wt             w       /   wt   \
    # Qb = b - 2*-------.b  = b - -----* 2*| ------.b |
    #            ||w||^2          ||w||     \ ||w||  /
    for i in range(col):
        j      =  i + 1
        Qb[i:] -= Rw[i:, i:j]@(2*(Rw[i:, i:j].T@Qb[i:]))

    return Qb


def householderSolution(Rw, Rdiag, singularity, b):
    if singularity is True: 
        print("\nMatrix is singular!!!\n")
        return None
    m, n = np.shape(Rw)
    if m < n: # fat matrix
        # 2. forward substitution: Rt*y = b
        y_hat = linearS.forwSub(Rw.T[:, :m], b, None, None, Rdiag)
        # 3. serial matrix-vector multiplications to y: x = (Qw1*...*Qwm)*y
        x_hat = householderRHSfat(Rw, y_hat, m, n)
    else:     # square or tall matrix
        # 2. serial matrix-vector multiplications to b: Qtb = (Qwn*(...*(Qw1*b)))
        Qtb   = householderRHStall(Rw, b, m, n)
        # 3. back substitution: R*x = Qtb
        x_hat = linearS.backSub(Rw[:n, :], Qtb[:n], None, None, Rdiag)

    return x_hat


def householderSolve(A, b):
    m, n = np.shape(A)
    if m < n: # fat matrix
        # 1. QR decomposition (composing R and obtaining w/||w|| for later use) 
        # Rw: R matrix filled with w/||w|| on the zero elements at the bottom left zone
        # Rdiag: vector of the diagonals of the R matrix
        Rw, Rdiag, singularity = householderDecomposition(A.T, n, m)
        if singularity is True: 
            print("\nMatrix is singular!!!\n")
            return None, Rw, Rdiag
        # 2. forward substitution: Rt*y = b
        y_hat                  = linearS.forwSub(Rw.T[:, :m], b, None, None, Rdiag)
        # 3. serial matrix-vector multiplications to y: x = (Qw1*...*Qwm)*y
        x_hat                  = householderRHSfat(Rw, y_hat, m, n)
    else:     # square or tall matrix
        # 1. QR decomposition (composing R and obtaining w/||w|| for later use) 
        Rw, Rdiag, singularity = householderDecomposition(A, m, n)
        if singularity is True: 
            print("\nMatrix is singular!!!\n")
            return None, Rw, Rdiag
        # 2. serial matrix-vector multiplications to b: Qtb = (Qwn*(...*(Qw1*b)))
        Qtb                    = householderRHStall(Rw, b, m, n)
        # 3. back substitution: R*x = Qtb
        x_hat                  = linearS.backSub(Rw[:n, :], Qtb[:n], None, None, Rdiag)

    return x_hat, Rw, Rdiag


# A_hat = Q*R = Q1*Q2*...*Qn*R
def AhatHouseholder(Rw, Rdiag):
    (m, n)                          = Rw.shape
    R                               = copy.deepcopy(Rw)
    R[np.tril_indices_from(Rw, -1)] = 0
    np.fill_diagonal(R, Rdiag)
    QtR                             = copy.deepcopy(R)

    if m > n: col = n       # tall matrix
    else:     col = n - 1   # square matrix

    # from column n-1 back to 0
    for i in range(col-1, -1, -1):
        if Rdiag[i] == 0: continue
        j        =  i + 1
        QtR[i:, :] -= Rw[i:, i:j]@(2*(Rw[i:, i:j].T@QtR[i:, :]))

    return QtR, R


if __name__ == "__main__":
    # A = np.array([[3, 5], [2, 2], [0, 0], [5, 4]], dtype=float)
    # A = np.array([[3, 14], [2, 70], [0, 0], [6, 35]], dtype=float)
    # A = np.array([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]], dtype=float)
    # A = np.array([[-1, 1], [2, 4], [-2, -1]], dtype=float)
    A = np.array([[0, 1], [np.log(2), 1], [np.log(3), 1], [np.log(4), 1], [np.log(5), 1]], dtype=float)
    b = np.array([1, 3.8, 5.9, 7, 8], dtype=float)
    # b = np.ones(A.shape[0], dtype=float)
    # x = np.ones(A.shape[1], dtype=float)
    # b = A@x
    
    xhat, Rw, Rdiag = householderSolve(A, b)
    print(f"\nxhat:\n{xhat}\nR:\n{Rw}\nRdiag:\n{Rdiag}")

    QRhouseholder, Rhat   = AhatHouseholder(Rw, Rdiag)
    np.set_printoptions(suppress=True)
    print(f"\nAhat:\n{QRhouseholder}\n")
    print(f"{Qmatrix(Rw, A.shape[0], A.shape[1])}")



# %%
