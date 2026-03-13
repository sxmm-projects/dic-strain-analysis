# %%
import copy
import time
import numpy                                                             as np
import scipy.linalg                                                      as solver
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'L3'))
import L3_c1_linearSolve                                                 as linearS
from   L4_c3_QRhouseholderReflection import reflectionPlaneUnitNormalVec as Hreflect

eps = linearS.eps

# Step 1: matrix upperbidiagonalization using alternating column and row Householder reflections
#         + accumulation of orthogonal matrices Ut and V
# B = UtAV
def householderBidiagonal(A, m, n, tol=eps):
    d    = np.zeros(n, dtype=float)     # diagonal vector
    f    = np.zeros(n-1, dtype=float)   # superdiagonal vector
    ww   = copy.deepcopy(A)             # w/||w|| of both column (left) and row (right) reflections
    V    = np.identity(n, dtype=float)  # V  = I
    Ut   = np.identity(m, dtype=float)  # Ut = I
    r    = n - 2                        # no. of row (right) Householder reflections (less than n by 2)
    stop = 0.0                          # ||B||inf: iteration stopping criterion for the next step (diagonalization)

    if m > n: col = n                   # tall matrix
    else:     col = n - 1               # square matrix

    # Step 1: matrix upperbidiagonalization using alternating column and row Householder reflections
    for i in range(col):
        # 1.1 column (left) Householder reflection to remove elements below the diagonal of column i
        ip1  = i + 1
        wi   = ww[i:, i:ip1]            # column vector y stored in the form of a matrix with 1 column
        # computing wi/||wi|| + storing it in column i of ww; storing the diagonal of B in d
        d[i] = Hreflect(wi, d[i])
        if d[i]:                        # d != 0 (nonzero diagonal)
            # B = QA = A - 2*w*wt*A/(wt.w)
            Bi = ww[i:, ip1:]
            if Bi.size: Bi -= wi@(2*(wi.T@Bi))
        dpf = np.abs(d[i])              # dpf = |d[i]| + |f[i]|

        # 1.2 row (right) Householder reflection to remove elements after the superdiagonal of row i if i < r
        if i < r:
            # wi
            wi   = ww[i:ip1, ip1:].T    # column vector y stored in the form of a matrix with 1 column
            # computing wi/||wi|| + storing it in row i of ww; storing the superdiagonal of B in f
            f[i] = Hreflect(wi, d[i])
            if f[i]:                    # f != 0 (nonzero superdiagonal)
                # B = AQ = A - 2*A*w*wt/(wt.w)
                ww[ip1:, ip1:] -= (2*(ww[ip1:, ip1:]@wi))@wi.T
                dpf += np.abs(f[i])     # dpf = |d[i]| + |f[i]|
        # 1.2 assigning B[n-2, n-1] to the last superdiagonal f[n-2]
        elif f[i:].size > 0:
            f[i] =  ww[i, ip1]
            dpf  += np.abs(f[i])        # dpf = |d[i]| + |f[i]|
        
        if dpf > stop: stop = dpf       # stop = max (|d[i]| + |f[i]|)
                                        #         i
    # 1.1 for square matrix: assigning B[n-1, n-1] to the last diagonal d[n-1]
    if m == n: d[col] = ww[col, col]   

    # Step 2: accumulation of right orthogonal matrices from right to left 
    #         (multiplication of Hhat to the left side of V => less flops)
    # V = Hhat0*Hhat1*...*Hhatn-3 = Hhat0*(Hhat1*(...*Hhatr-1))
    #           /       w*wt \           /  w  \   /   wt  \
    # Hhat*V = |I - 2*------- |*V = V - | ----- |*|2*-----*V|
    #           \     ||w||^2/           \||w||/   \ ||w|| /
    for i in range(r-1, -1, -1):
        ip1 = i + 1
        if f[i]:                        # nonzero superdiagonal f[i]
            V[ip1:, ip1:] -= ww[i:ip1, ip1:].T@(
                             2*(ww[i:ip1, ip1:]@V[ip1:, ip1:]))

    # Step 3: accumulation of left orthogonal matrices from left to right
    #         (multiplication of H to the right side of Ut => less flops)
    # Ut = Hn-1*...*H1*H0 = ((Hn-1*...)*H1)*H0
    #            /       w*wt \          /      w  \   /  wt \
    # Ut*H = Ut*|I - 2*------- | = Ut - |2*Ut*----- |*| ----- |
    #            \     ||w||^2/          \    ||w||/   \||w||/
    for i in range(col-1, -1, -1):
        ip1 = i + 1
        if d[i]:                        # nonzero diagonal d[i]
            Ut[i:, i:] -= (2*(Ut[i:, i:]@ww[i:, i:ip1]))@ww[i:, i:ip1].T

    #     V = Hhatnxn, Ut = Hmxm
    return V, d, Ut, f, tol*stop        # tol depending on both computing platform (eps)
                                        #     & matrix element values (||B||inf)


# Step 2: diagonalization using implicit shift symmetric QR transformation by iterative Givens rotations
#         + accumulation of orthogonal matrices Ut and V
def diagonalization(n, V, d, Ut, f, stop):
    #                                                                      mixed biadiagonal/diagonal
    #       _    k1                    k2                                 _ submatrix; f[k1-1] == 0
    #      |B1 |                            |                              | B1 = B[0:k1, 0:k1]
    #      |___|____________________________|______________________________|
    #  k1->|   |d[k1] f[k1]!=0              |                              |full bidiagonal submatrix
    #      |   |       .        .           |                              | B2 = B[k1:k2+1, k1:k2+1]
    # B =  |   |          d[k2-1] f[k2-1]!=0|                              |    f[k1:k2-1] != 0
    #  k2->|___|_____________________d[k2]__|f[k2]==0______________________|         f[k2] == 0
    #      |   |                            | d[k2+1] f[k2+1]==0           |
    #      |   |                            |          .        .          |  diagonal submatrix
    #      |   |                            |              d[n-2] f[n-2]==0| B3 = B[k2+1:n, k2+1:n]
    #      |_  |                            |                       d[n-1]_|   f[k2+1:n-1] == 0
    Gi    = np.zeros((2, 2), dtype=float)
    nm1   = n - 1
    k2    = nm1
    rank  = n
    itera = 0
    
    # expanding B3 bottom up until B3 == B (full diagonal matrix)
    #  -> reducing k2 until k2 == 0 when f[:] == 0
    while k2 > 0: 
        itera += 1
        # Special step if d[n-1] == 0 (singular matrix): squeezing from the bottom right
        # zeroing f[n-2] by successive Givens column rotations multiplied to the right 
        if (k2 == nm1) and (np.abs(d[nm1]) < stop):
            d[nm1] = 0
            #  _     _        _             _   _        _     _                      _
            # |       |*Gi = | f[i-2]    0   |*| cos -sin | = | f[i-2]*cos -f[i-2]*sin |
            # |_xa xb_|      |_d[i-1] f[i-1]_| |_sin  cos_|   |_    r           0     _|
            k, Gi[0, 0], Gi[0, 1] = 1, 0, 1
            for i in range(nm1, 0, -1): 
                p1           =  i - 1
                xb           =  f[p1]*Gi[0, 1]  # xb     = f[i-1]*(-sin)
                f[p1]        *= Gi[0, 0]        # f[i-1] = f[i-1]*cos
                # xb == 0 -> the whole last column B[:, n-1] == 0 -> stop
                if np.abs(xb) < stop: break
                # computing Gi for the next rotation and accumulatively updating V
                xa           =  d[p1]
                r            =  d[p1]    = np.sqrt(xa*xa + xb*xb)
                Gi[0, 0]     =  Gi[1, 1] = xa/r # cos
                Gi[1, 0]     =  xb/r            # sin
                Gi[0, 1]     =  -Gi[1, 0]       # -sin
                V[:, p1:n:k] =  V[:, p1:n:k]@Gi
                k            += 1

        # 1. B split into B1, B2 and B3 by searching for k1 (k2 already known)
        k2m1  = k2 - 1
        split = False
        for i in range(k2m1, -1, -1):
            # f[i] == 0
            if np.abs(f[i]) < stop:
                f[i]  = 0
                split = True
                break
            # Special step if d[i] == 0 (singular matrix): squeezing from the top left
            # d[i] == 0 -> zeroing f[i]
            if np.abs(d[i]) < stop:
                d[i] = 0
                # zeroing f[i] by succesive Givens row rotations multiplied to the left
                #     _    _     _         _   _             _     _            _
                # Gi*| xa   | = |  cos  sin |*|  f[i]     0   | = | 0 sin*f[i+1] |
                #    |_xb  _|   |_-sin  cos_| |_d[i+1] f[i+1]_|   |_r cos*f[i+1]_|
                k, Gi[0, 0], Gi[0, 1] = 1, 0, 1
                for j in range(i, k2):
                    p1   =  j + 1
                    xa   =  Gi[0, 1]*f[j]
                    f[j] *= Gi[0, 0]
                    # xa == 0 -> the whole row j B[j, :] == 0 -> stop
                    if np.abs(xa) < stop: break
                    xb              =  d[p1]
                    r               =  d[p1]    = np.sqrt(xa*xa + xb*xb)
                    Gi[0, 0]        =  Gi[1, 1] = xb/r  # cos
                    Gi[0, 1]        =  -xa/r           # sin
                    Gi[1, 0]        =  -Gi[0, 1]       # -sin
                    Ut[i:p1+1:k, :] =  Gi@Ut[i:p1+1:k, :]
                    k               += 1
                split = True
                break
        if split:         # zero-superdiagonal found at i
            if i == k2m1: # i == k2 - 1 (just above row k2)
                # making the final d[k2] positive (no longer touched)
                # singular value must be >= 0
                if d[k2] == 0: rank -= 1
                elif d[k2] < 0:
                    d[k2]    = -d[k2]
                    V[:, k2] = -V[:, k2]
                k2 = k2m1 # k2 = k2 - 1 (reducing k2 by 1)
                if not k2:# k2 == 0
                    if d[k2] == 0: rank -= 1
                    elif d[k2] < 0:
                        d[k2]    = -d[k2]
                        V[:, k2] = -V[:, k2]
                continue  # resplitting B after k2 update
            k1 = i + 1    # set k1 to the row just below row i where f[k1] != 0
        else: k1 = 0      # no zero-superdiagonal found -> k1 = 0

        # 2. reducing superdiagonals f[k1:k2] in submatrix B[k1:k2,k1:k2+1] 
        #    hopefully to zero by symmetric Given rotations implicitly on 
        #    symmetric, tridiagonal BT*B (GT*BT*B*G)
        #                                                     
        #           _      k1                                                 k2      _
        #      k1->|  d[k1]*d[k1]    d[k1]*f[k1]                                       |
        #          |                                                                   |
        #          |  d[k1]*f[k1]  d[k1+1]*d[k1+1]       .                             |
        #          |                +f[k1]*f[k1]                                       |
        #          |           .                    .                   .              |
        # B2T*B2 = |                               _                                   |______
        #          |                              |  d[k2-1]*d[k2-1]   d[k2-1]*f[k2-1] |
        #          |                        .     | +f[k2-2]*f[k2-2]                   |
        #          |                              |                                    | = T
        #          |                              |  d[k2-1]*f[k2-1]     d[k2]*d[k2]   |
        #      k2->|_                             |_                  +f[k2-1]*f[k2-1]_|______      
        # 
        # 2.1 determination of the eigenvalue of T (2x2 bottom right matrix of B2) closer to
        #     B2T*B2[k2, k2] = Wilkinson shift, shifting T[:, 0] for faster convergence
        #                                                                          
        #       T00 - T11          ___________                                    
        #  xa = --------- , xb = \/ xa*xa + 1  
        #         2*T01                                          
        #                                       ________________________
        #                                      /           2          2
        #                    (T00 - T11) +|- \/ (T00 - T11)  + (2*T01)    
        #   r = xa +|- xb = --------------------------------------------- 
        #                                      2*T01                      
        #                           ________________________ 
        #                          /           2          2                      
        #        (T00 - T11) +|- \/ (T00 - T11)  + (2*T01)      conj   
        #     = --------------------------------------------- * ---- 
        #                           2*T01                       conj                     
        #                          2               2          2   
        #               (T00 - T11)  -  (T00 - T11)  - (2*T01)                        
        #     = -------------------------------------------------------- 
        #               _                    ________________________ _                    
        #        2*T01*|                    /           2          2   |                     
        #              |_ (T00 - T11) +|- \/ (T00 - T11)  + (2*T01)   _|   
        #
        #                          -2*T01                           -1
        #     = --------------------------------------------- = -----------
        #                           ________________________     xa +|- xb
        #                         /           2          2
        #       (T00 - T11) +|- \/ (T00 - T11)  + (2*T01)
        #
        #                                     ________________________
        #                                    /           2          2
        #                  (T00 + T11) +|- \/ (T00 - T11)  + (2*T01) 
        #  lamb1, lamb2 = --------------------------------------------- = r*T01 + T11
        #                                      2
        #                           T01
        #               = T11 - ----------- 
        #                        xa +|- xb
        # lamb closest to T11 if the second term is min. and hence xa +|- xb is max.
        # -> 1. if xa < 0, xa - xb, 2. if xa > 0, xa + xb    (xb always positive)
        Gi[0, 0] = d[k2m1]*d[k2m1]                    # T00
        if (k2 - k1) > 1: Gi[0, 0] += f[k2-2]*f[k2-2] # if B2 = 2x2 -> B2[0,0] = d*d
        Gi[1, 1] =     d[k2]*d[k2] + f[k2m1]*f[k2m1]  # T11
        Gi[0, 1] = d[k2m1]*f[k2m1]                    # T01
        xa       = (Gi[0, 0] - Gi[1, 1])/(2*Gi[0, 1])
        xb       = np.sqrt(xa*xa + 1)
        # xa/d[k1] = (d[k1]^2 - lamb)/d[k1]
        # so that cos = (xa/d[k1])/(r/d[k1]) = xa/r
        #         sin = xb/r = f[k1]/(r/d[k1]) = d[k1]*f[k1]/r
        ya       = d[k1]
        xa       = (ya*ya - ((Gi[1, 1] - Gi[0, 1]/(xa - xb)) 
                             if xa < 0 else (Gi[1, 1] - Gi[0, 1]/(xa + xb))))/ya
        Gi[0, 0] = Gi[0, 1] = 1 # cos = sin = 1

        for i in range(k1, k2):
            p1 = i + 1
            # 2.2 Givens column rotation matrix multiplied to the right
            # 2.2.1 G(1, 2): matrix shifted by lamb when i = k1
            #  _                               _   _        _     _         _
            # |_ d[k1]*d[k1]-lamb  d[k1]*f[k1] _|*| cos -sin | = |_  r   0  _|
            #                                     |_sin  cos_|  
            #
            # xa = d[k1]*d[k1] - lamb, xb = d[k1]*f[k1]
            # -> r = sqrt(xa^2 + xb^2), cos = xa/r, sin = xb/r
            #  _             _   _        _     _                                         _
            # |   0      0    | | cos -sin |   |         0                      0          |
            # | d[k1]  f[k1]  |*|_sin  cos_| = | d[k1]*cos+f[k1]*sin  -d[k1]*sin+f[k1]*cos |
            # |_  0   d[k1+1]_|                |_    d[k1+1]*sin           d[k1+1]*cos    _|
            #
            # 2.2.2 G(i, i+1): i = 2 - k2 (no shift)
            #  _           _   _        _     _                                    _  
            # | xa xb=s*f[i]| | cos -sin |   |     f[i-1] = r             0         |
            # | ya yb=c*f[i]|*|_sin  cos_| = |  xa=ya*cos+yb*sin  ya=-ya*sin+yb*cos |
            # |_ 0  d[i+1] _|                |_ xb=d[i+1]*sin     yb=d[i+1]*cos    _|  
            #                                                                            
            # cos and sin from previous left matrix in 2.3 of the previous loop (i - 1)
            # xa and ya already calculated in 2.3 (i - 1)
            # xb = sin*f[i], yb = cos*f[i]
            # -> r = sqrt(xa^2 + xb^2) of current right matrix here in 2.2
            # -> cos = xa/r, sin = xb/r  of current right matrix here in 2.2
            xb           = Gi[0, 1]*f[i]           # sin*f[i] 
            yb           = Gi[0, 0]*f[i]           # cos*f[i] 
            r            = np.sqrt(xa*xa + xb*xb)  # xa from previous 2.3
            if i > k1: f[i-1] = r                  # f[k1-1] above B2 == 0 or non-existent
            Gi[0, 0]     = Gi[1, 1] = xa/r         # cos 
            Gi[1, 0]     = xb/r                    # sin
            Gi[0, 1]     = -Gi[1, 0]               # -sin
            V[:, i:p1+1] = V[:, i:p1+1]@Gi
            xa           = ya*Gi[0, 0] + yb*Gi[1, 0]
            ya           = ya*Gi[0, 1] + yb*Gi[1, 1]
            xb           = d[p1]*Gi[1, 0]
            yb           = d[p1]*Gi[1, 1]
                                                                              
            # 2.3 Givens row rotation matrix multiplied to the left                             
            #     G(i, i+1): i = 1 - k2 (no shift)                                              
            #  _        _   _            _     _                                            _   
            # |  cos sin |*| xa ya   0    | = |  d[i] = r  xa=cos*ya+sin*yb   xb=sin*f[i+1]  |
            # |_-sin cos_| |_xb yb f[i+1]_|   |_     0     ya=-sin*ya+cos*yb  yb=cos*f[i+1] _|
            #
            # xa, xb, ya and yb are all already prepared from 2.2
            # -> r = sqrt(xa^2 + xb^2) of current left matrix here in 2.3
            # -> cos = xa/r, sin = xb/r  of current left matrix here in 2.3
            d[i]          = r        = np.sqrt(xa*xa + xb*xb)
            Gi[0, 0]      = Gi[1, 1] = xa/r        # cos 
            Gi[0, 1]      = xb/r                   # sin
            Gi[1, 0]      = -Gi[0, 1]              # -sin
            Ut[i:p1+1, :] = Gi@Ut[i:p1+1, :]
            xa            = Gi[0, 0]*ya + Gi[0, 1]*yb
            ya            = Gi[1, 0]*ya + Gi[1, 1]*yb
            # xb and yb calculated in the next loop i + 1
        f[k2m1] = xa
        d[k2]   = ya

    return rank, itera


# Step 3: permutation
def matPermutation(n, V, d, Ut):
    for i in range(n - 1):
        maxID = np.argmax(d[i:]) + i
        if maxID == i: continue
        d[i], d[maxID]    = d[maxID], d[i]
        Ut[[i, maxID], :] = Ut[[maxID, i], :]
        V[:, [i, maxID]]  = V[:, [maxID, i]]
    return


def SVDdecomposition(A):
    m, n              = np.shape(A)
    # Step 1: upper bidiagonalization using alternating column and row Householder reflections
    #         + accumulation of orthogonal matrices Ut and V
    V, d, Ut, f, stop = householderBidiagonal(A, m, n, eps)
    # Step 2: diagonalization using implicit shift symmetric QR transformation by iterative Givens rotations
    #         + accumulation of orthogonal matrices Ut and V where tol = eps*||B||inf
    rank, itera       = diagonalization(n, V, d, Ut, f, stop)
    # Step 3: accumulation of orthogonal matrices and permutation
    matPermutation(n, V, d, Ut)

    return V, d, Ut, rank, f, itera


def SVDsolve(V, d, Ut, rank, b):
    return (V[:, :rank]*(1.0/d[:rank]))@(Ut[:rank, :]@b)


# A_hat = U*Sig*Vt
def AhatSVD(V, d, Ut):
    D = np.zeros((V.shape[1], Ut.shape[1]), dtype=float)
    np.fill_diagonal(D, d)

    return (V@D@Ut).T


if __name__ == "__main__":
    # A = np.array([[3, 5], [2, 2], [0, 0], [5, 4]], dtype=float)
    # A = np.array([[3, 14], [2, 70], [0, 0], [6, 35]], dtype=float)
    A = np.array([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]], dtype=float)
    # A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
    # A = np.array([[16, 52, 80], [44, 80, -32], [-9, -36, -72], [-16, -16, 64]], dtype=float)/27
    # b = np.array([4, 1, 3, 0], dtype=float)
    # A = np.array([[-1, 1], [2, 4], [-2, -1]], dtype=float)
    # A = np.array([[1/10, 1/3, 0], [2/10, 2/3, 3], [3/10, 3/3, 0], [4/10, 4/3, 7]], dtype=float)
    # A[0, 0] += 10*eps
    # b = np.ones(A.shape[0], dtype=float)
    x = np.ones(A.shape[1], dtype=float)
    b = A@x
    
    V, d, Ut, rank, f, itera = SVDdecomposition(A)
    xhat                     = SVDsolve(V, d, Ut, rank, b)
    print(f"\nxhat:\n{xhat}\nV:\n{V}\nd:\n{d}\nUt:\n{Ut}\n")

    SVDAhat = AhatSVD(V, d, Ut)
    print(f"\nAhat:\n{SVDAhat}\nb:\n{b}\n")


# %%
