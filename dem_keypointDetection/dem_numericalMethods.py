# %%
import numpy                         as     np
import copy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'L3'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'L4'))
from   L3_c1_linearSolve             import eps, backSub, gaussElimination, LUdecomposition, LUsolve
from   L4_c2_QRgivensRotation        import givensRHStall, givensDecomposition
from   L4_c3_QRhouseholderReflection import householderSolve
from   L4_c4_SVD                     import SVDdecomposition

    
# CHAPTER 7: nonlinear root finding (damped Newton Rahpson)
def dampedNr(data, tol=1e-5, zmax=100, lambmin=0.001):
    z     = 0
    x, f  = data.getData(), data.errorFunction()
    norme = tol*10
    
    while norme > tol and z < zmax:
        diff    = data.dFx(f)
        lu, per = LUdecomposition(diff, True)
        s       = LUsolve(lu, per, -f)         # s = x(z+1) - x(z)
        norms   = np.sqrt(np.dot(s, s))
        
        lamb    = 1.0
        C       = 0.75
        norme   = norms
        while norme > C*norms:
            C     =  1.0 - lamb/4.0
            xnew  =  x + lamb*s
            data.updateData(xnew)
            fnew  =  data.errorFunction()
            e     =  LUsolve(lu, per, -fnew) # s = x(z+1) - x(z)
            norme =  np.sqrt(np.dot(e, e))
            lamb  /= 2.0
            if lamb < lambmin:
                if lambmin < 1:
                    print(f"\ninitial guess is too far from a root x*.\n",
                          f"===> (lambda = {lamb} < {lambmin}) please reconsider x0 <===\n") 
                break

        x, f  =  xnew, fnew
        z     += 1

    print(f"\t{z:>3d}.\tf = {f}\tx = {x}\terror = {norme}")

    return z


# CHAPTER 8: nonlinear curve fitting (Levenberg-Marquardt method)
def levenbergMarquardtNormal(data, mu=1.0, b0=0.1, b1=0.9, zmax=100, cmax=5, dmax=2, tol=1e-5):
    if b0 > b1:   b0, b1 =  b1, b0
    if b0 < 0:    b0     =  0.1
    if b1 > 1:    b1     =  0.9
    if mu <= 0.0: mu     =  1.0

    lim                  =  eps
    muLim                =  np.sqrt(lim)
    tol                  *= tol
    z                    =  0
    d                    =  1
    x                    =  data.getData()
    fx, normF2           =  data.errorFunction()
    A, b, diff, normPhi2 =  data.dFx(fx)
    n                    =  x.size  # number of variables
    
    while normPhi2 > tol:
        if z == zmax:
            if d >= dmax: break
            print(f"\t{d:>3d}.\t{z:>3d} iters\tmu = {mu:.2e}\terror = {np.sqrt(normF2):.2e}"
                  f"\tslope = {np.sqrt(normPhi2):.2e}")
            z  =  0
            d  += 1
            mu *= np.sqrt(tol)/d
            if mu < muLim: mu = muLim

        rho =  0.0
        mu  /= 2.0
        c   =  0

        while rho < b0 and c < cmax:
            c                  += 1
            # too small rho (too little damping) => diverging => increasing damping factor mu
            mu                 *= 2.0       
            # CHAPTER 2, 3: linear equations and curve fitting (normal equation + LDLT or LU decomposition)
            #                                             adding the damping factor to A
            s                  =  data.normalEqationsSolve(A + np.diag(np.full(n, mu*mu)), b)
            if s is None:
                muLim *= 10
                mu    *= 10
                continue
            xnew               =  x + s
            data.updateData(xnew)
            fxnew, normFnew2   =  data.errorFunction()

            # subtractive cancellation when calculating rho if normPhi2 is very small
            # (normF2 ~ normFnew2 ~ rho) => infinite rho 
            if normPhi2 > lim:  
                tmp = normF2 - normFnew2
                if tmp < 0:
                    rho = 0 
                    continue
                rho = fx + diff@s
                rho = np.dot(rho, rho)
                rho = np.inf if normF2 == rho else tmp/(normF2 - rho)
            else: break

        # too big rho (too much damping) => slowly converging => lowering damping factor mu
        # too small mu leading to singular R matrix => no further decrease of mu
        if rho > b1 and mu > muLim: mu /= 3.0

        x                    = xnew
        fx                   = fxnew
        normF2               = normFnew2 
        A, b, diff, normPhi2 = data.dFx(fx)
        z                    = z + 1

    print(f"\t{d:>3d}.\t{z:>3d} iters\tmu = {mu:.2e}\terror = {np.sqrt(normF2):.2e}\tslope = {np.sqrt(normPhi2):.2e}")

    return d*z

# CHAPTER 8: nonlinear curve fitting (Levenberg-Marquardt method)
def levenbergMarquardt(data, mu=1.0, b0=0.1, b1=0.9, zmax=100, cmax=5, dmax=2, tol=1e-5):
    if b0 > b1:   b0, b1 =  b1, b0
    if b0 < 0:    b0     =  0.1
    if b1 > 1:    b1     =  0.9
    if mu <= 0.0: mu     =  1.0

    lim            =  eps
    muLim          =  np.sqrt(lim)
    tol            *= tol
    z              =  0
    d              =  1
    x              =  data.getData()
    fx, normF2     =  data.errorFunction()
    diff, normPhi2 =  data.dFx(fx)
    m              =  fx.size # number of data points
    n              =  x.size  # number of variables
    diffext        =  np.zeros((m + n, n), dtype=float)
    fext           =  np.zeros(m + n, dtype=float)
    
    while normPhi2 > tol:
        if z == zmax:
            if d >= dmax: break
            print(f"\t{d:>3d}.\t{z:>3d} iters\tmu = {mu:.2e}\terror = {np.sqrt(normF2):.2e}"
                  f"\tslope = {np.sqrt(normPhi2):.2e}")
            z  =  0
            d  += 1
            mu *= np.sqrt(tol)/d
            if mu < muLim: mu = muLim

        rho =  0.0
        mu  /= 2.0
        c   =  0

        while rho < b0 and c < cmax:
            c                += 1
            # too small rho (too little damping) => diverging => increasing damping factor mu
            mu               *= 2.0       
            # 1. filling the upper block
            diffext[:m, :]   =  diff[:, :]
            fext[:m]         =  fx[:]
            # 2. filling the lower block 
            diffext[m:, :]   =  fext[m:] = 0     # all elements in the lower block are zero
            np.fill_diagonal(diffext[m:, :], mu) # extended damping
            # CHAPTER 4: linear curve fitting (QR decomposition)
            s                =  householderSolve(diffext, -fext)[0]
            if s is None:
                muLim *= 10
                mu    *= 10
                continue
            xnew             =  x + s
            data.updateData(xnew)
            fxnew, normFnew2 =  data.errorFunction()

            # subtractive cancellation when calculating rho if normPhi2 is very small
            # (normF2 ~ normFnew2 ~ rho) => infinite rho 
            if normPhi2 > lim:  
                tmp = normF2 - normFnew2
                if tmp < 0:
                    rho = 0 
                    continue
                rho = fx + diff@s
                rho = np.dot(rho, rho)
                rho = np.inf if normF2 == rho else tmp/(normF2 - rho)
            else: break

        # too big rho (too much damping) => slowly converging => lowering damping factor mu
        # too small mu leading to singular R matrix => no further decrease of mu
        if rho > b1 and mu > muLim: mu /= 3.0

        x              = xnew
        fx             = fxnew
        normF2         = normFnew2 
        diff, normPhi2 = data.dFx(fx)
        z              = z + 1

    print(f"\t{d:>3d}.\t{z:>3d} iters\tmu = {mu:.2e}\terror = {np.sqrt(normF2):.2e}\tslope = {np.sqrt(normPhi2):.2e}")

    return d*z

# CHAPTER 6: nonlinear curve fitting (Levenberg-Marquardt method)
#               _      _                  _   _
#              |   A    | =              |  b  |
#              |_ mu*I _|                |_ 0 _|
#  _        _   _      _     _        _   _   _  
# |  QwT  0  |*|   A    | = |  QwT  0  |*|  b  | 
# |_  0   I _| |_ mu*I _|   |_  0   I _| |_ 0 _|
# 
# 1. Householder reflection QR decomposition on A and b (upper blocks)
#              _       _           _       _ 
#             |  QwT*A  | =       |  QwT*b  | 
#             |_  mu*I _|         |_   0   _|
#              _       _           _       _ 
#             |    Rw   | =       |  QwT*b  | 
#             |_  mu*I _|         |_   0   _|
# 
# 2. Givens rotation QR dceomposition on the whole system
#               _      _           _       _ 
#        QgT * |   Rw   | = QgT * |  QwT*b  |
#              |_ mu*I _|         |_   0   _|
#                  _   _           _       _ 
#                 |  R  | = QgT * |  QwT*b  |
#                 |_ 0 _|         |_   0   _|
# 
def levenbergMarquardtBlock(data, mu=1.0, b0=0.1, b1=0.9, zmax=100, cmax=5, dmax=2, tol=1e-5):
    if b0 > b1:   b0, b1 =  b1, b0
    if b0 < 0:    b0     =  0.1
    if b1 > 1:    b1     =  0.9
    if mu <= 0.0: mu     =  1.0

    lim            =  eps
    muLim          =  np.sqrt(lim)
    tol            *= tol
    z              =  0
    d              =  1
    x              =  data.getData()
    fx, normF2     =  data.errorFunction()
    diff, normPhi2 =  data.dFx(fx)
    m              =  fx.size # number of data points
    n              =  x.size  # number of variables
    Qb             =  np.zeros(m + n, dtype=float)
    R              =  np.zeros((m + n, n), dtype=float)
    elRange        =  np.array([np.full(n, m), np.arange(m+1, m+n+1)], dtype=int).T
    
    while normPhi2 > tol:
        if z == zmax:
            if d >= dmax: break
            print(f"\t{d:>3d}.\t{z:>3d} iters\tmu = {mu:.2e}\terror = {np.sqrt(normF2):.2e}"
                  f"\tslope = {np.sqrt(normPhi2):.2e}")
            z  =  0
            d  += 1
            mu *= np.sqrt(tol)/d
            if mu < muLim: mu = muLim

        rho   =  0.0
        mu    /= 2.0
        c     =  0
        # CHAPTER 4: linear curve fitting (QR decomposition)
        # 1st QR dcomposition to obtain the upper Rw block and b
        Rw, b =  data.upperBlockQRdecomposition(diff, -fx, m, n)

        while rho < b0 and c < cmax:
            c                += 1
            # too small rho (too little damping) => diverging => increasing damping factor mu
            mu               *= 2.0
            # 1. filling the upper block R[:m, :] and Qb[:m] obtained from the 1st QR decomposition
            Qb[:m]           =  b[:]
            R[:m, :]         =  Rw[:, :]
            Qb[m:]           =  R[m:, :] = 0 # all elements in the lower block are zero
            # 2. 2nd QR decomposition (Givens rotation) to remove mu*I elements in the lower block 
            np.fill_diagonal(R[m:, :], mu)   # extended damping
            R                =  givensDecomposition(R, m, n, elRange)
            Qb               =  givensRHStall(R, Qb, m, n, elRange)
            # 3. solving for s using backsubstitution on the upper triagular block
            mat              =  R[:n, :]
            s                =  backSub(mat, Qb[:n], None, None, mat)
            if s is None:
                muLim *= 10
                mu    *= 10
                continue
            xnew             =  x + s
            data.updateData(xnew)
            fxnew, normFnew2 =  data.errorFunction()

            # subtractive cancellation when calculating rho if normPhi2 is very small
            # (normF2 ~ normFnew2 ~ rho) => infinite rho 
            if normPhi2 > lim:  
                tmp = normF2 - normFnew2
                if tmp < 0:
                    rho = 0 
                    continue
                rho = fx + diff@s
                rho = np.dot(rho, rho)
                rho = np.inf if normF2 == rho else tmp/(normF2 - rho)
            else: break

        # too big rho (too much damping) => slowly converging => lowering damping factor mu
        # too small mu leading to singular R matrix => no further decrease of mu
        if rho > b1 and mu > muLim: mu /= 3.0

        x              = xnew
        fx             = fxnew
        normF2         = normFnew2 
        diff, normPhi2 = data.dFx(fx)
        z              = z + 1

    print(f"\t{d:>3d}.\t{z:>3d} iters\tmu = {mu:.2e}\terror = {np.sqrt(normF2):.2e}\tslope = {np.sqrt(normPhi2):.2e}")

    return d*z

