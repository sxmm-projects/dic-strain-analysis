# %%
import copy
import math
import sys

import cv2
import numpy as np
from scipy.sparse.linalg import eigs

import dem_numericalMethods as nuMet

Imgs = []
_g_min = 0
_g_max = 255
# optimal step size for the first derivative
_g_h   = math.pow(sys.float_info.epsilon, 1.0/3.0) 


def greyScale(img):
    greyImage = img.mean(2).round(decimals=0, out=None).astype('uint8')
    return greyImage


# CHAPTER 7: polynomial interpolation (power form)
#        .       .    |    .      .             .             .      .   |    .       .
#        .       .    |    .      .             .             .      .   |    .       .
#        .       .    |    .      .             .             .      .   |    .       .
# ... (-2,-2) (-2,-1) | (-2,0) (-2,1)          -2          (-2,1) (-2,0) | (-2,-1) (-2,-2) ...
# ... (-1,-2) (-1,-1) | (-1,0) (-1,1)          -1          (-1,1) (-1,0) | (-1,-1) (-1,-2) ...
# --------------------+--------------------------------------------------+-----------------------> i
# ...  (0,-2)  (0,-1) | (0,0)  (0,1)  (0,2)     0    (0,2)  (0,1)  (0,0) | (0,-1)  (0,-2)  ...
# ...  (1,-2)  (1,-1) | (1,0)  (1,1)            1           (1,1)  (1,0) | (1,-1)  (1,-2)  ... 
#                     | (2,0)                   2                  (2,0) |
#                     |                                                  |
#                     |                                                  |
# ...    -2      -1   |   0      1      2              2      1      0   |   -1      -2    ...
#                     |                                                  |
#                     |                                                  |
#                     | (2,0)                   2                  (2,0) |
# ...  (1,-2)  (1,-1) | (1,0)  (1,1)            1           (1,1)  (1,0) | (1,-1)  (1,-2)  ...
# ...  (0,-2)  (0,-1) | (0,0)  (0,1)  (0,2)     0    (0,2)  (0,1)  (0,0) | (0,-1)  (0,-2)  ...
# --------------------+--------------------------------------------------+-----------------------
# ... (-1,-2) (-1,-1) | (-1,0) (-1,1)          -1          (-1,1) (-1,0) | (-1,-1) (-1,-2) ...
# ... (-2,-2) (-2,-1) | (-2,0) (-2,1)          -2          (-2,1) (-2,0) | (-2,-1) (-2,-2) ...
#        .       .    |    .      .             .             .      .   |    .       .
#        .       .    |    .      .             .             .      .   |    .       .
#        .       .    |    .      .             .             .      .   |    .       .
#                     |
#                    \/
#                     j


#  _                            _   _  _     _   _
# | 1  i0  j0  i0*j0  i0^2  j0^2 | | a0 |   | I00 |
# | 1  i0  j1  i0*j1  i0^2  j1^2 | | a1 |   | I01 |
# | 1  i0  j2  i0*j2  i0^2  j2^2 |x| a2 | = | I02 |
# | 1  i1  j0  i1*j0  i1^2  j0^2 | | a3 |   | I10 |
# | 1  i1  j1  i1*j1  i1^2  j1^2 | | a4 |   | I11 |
# |_1  i2  j0  i2*j0  i2^2  j0^2_| |_a5_|   |_I20_|
# 
def cornerPolynomialCalc(I00, I01, I02, I10, I11, I20, x):
    a0  = I00
    a5  = (I02 - 2*I01 + I00)/2.0
    a4  = (I20 - 2*I10 + I00)/2.0
    a2  = I01 - I00 - a5
    a1  = I10 - I00 - a4
    a3  = I11 - a0 - a1 - a2 - a4 - a5

    i   = x[0]
    j   = x[1]
    res = a0 + (a1 + a4*i)*i + (a2 + a3*i + a5*j)*j

    return res

#  _           _   _  _     _  _
# | 1  x0  x0^2 | | a0 |   | I0 |
# | 1  x1  x1^2 |x| a1 | = | I1 |
# |_1  x2  x2^2_| |_a2_|   |_I2_|
#
# direction = False +--------------->
#           = True  +
#                   |
#                   |
#                  \/
# 
# inc = False {-1, -2, ..., -k}
#     = True  {-k, ..., -2, -1}
def sidePolynomialCalc(k, I0, I1, I2, direction=False, inc=False):
    a0  = I0
    a2  = (I2 - 2*I1 + I0)/2.0
    a1  = I1 - I0 - a2

    res = np.zeros((k, I0.size), dtype=I0.dtype)
    ran = range(-k, 0)
    if not inc: ran = np.flip(ran)
    j   = 0
    for i in ran:  # values not exceeding intensity max and min
        res[j, :] =  a0 + a1*i + a2*i*i
        j         += 1

    return res if not direction else res.T


def borderExtrapolation(imgRaw, k):
    img      = imgRaw.astype(float)
    N0, N1   = img.shape
    eN0, eN1 = tuple(map(sum, zip((N0, N1), (2*k, 2*k))))
    extImg   = np.zeros((eN0, eN1), dtype=float)
    b        = k + 2

    # 2nd-order polynomial extrapolation of the border of the original image img

    i                 = slice(b, eN1-b) # x range for the extended image
    j                 = slice(2, N1-2)  # x range for the original image
    extImg[:k, i]     = sidePolynomialCalc(k, img[0, j], img[1, j],
                                           img[2, j], False, True)        # top side
    extImg[eN0-k:, i] = sidePolynomialCalc(k, img[N0-1, j], img[N0-2, j],
                                           img[N0-3, j], False, False)    # bottom side


    i                 = slice(b, eN0-b) # y range for the extended image
    j                 = slice(2, N0-2)  # y range for the original image
    extImg[i, :k]     = sidePolynomialCalc(k, img[j, 0], img[j, 1],  
                                           img[j, 2], True, True)         # left side
    extImg[i, eN1-k:] = sidePolynomialCalc(k, img[j, N1-1], img[j, N1-2],
                                           img[j, N1-3], True, False)     # right side

    # top-left corner
    x                        = np.meshgrid(np.arange(-k, 2), np.arange(-k, 2), indexing='ij')
    extImg[:b, :b]           = cornerPolynomialCalc(img[0, 0], img[0, 1], img[0, 2],
                                                    img[1, 0], img[1, 1], img[2, 0], x)
    # top-right corner
    x                        = (x[0], np.flip(x[1], 1))
    extImg[:b, eN1-b:]       = cornerPolynomialCalc(img[0, N1-1], img[0, N1-2], img[0, N1-3],
                                                    img[1, N1-1], img[1, N1-2], img[2, N1-1], x)
    # bottom-right corner
    x                        = (np.flip(x[0], 0), x[1])
    extImg[eN0-b:, eN1-b:]   = cornerPolynomialCalc(img[N0-1, N1-1], img[N0-1, N1-2],
                                                    img[N0-1, N1-3], img[N0-2, N1-1],
                                                    img[N0-2, N1-2], img[N0-3, N1-1], x)
    # bottom-left corner
    x                        = (x[0], np.flip(x[1], 1))
    extImg[eN0-b:, :b]       = cornerPolynomialCalc(img[N0-1, 0], img[N0-1, 1], img[N0-1, 2], 
                                                    img[N0-2, 0], img[N0-2, 1], img[N0-3, 0], x)
    
    # copy the inner original image
    extImg[k:eN0-k, k:eN1-k] = img  

    return extImg.astype(int)

def imgConvolution(img, filterKernel):
    n = filterKernel[0].shape
    if len(n) != 2: 
        print("False kernel dimensions!!")
        return None
    elif n[0] != n[1]:
        print("Non-square kernel shape!!")
        return None
    elif (n[0] % 2) < 1:
        print("Even kernel width or length!!")
        return None

    N0, N1  = img.shape
    extImg  = borderExtrapolation(img, int(n[0]/2))        # extending the image borders
    fKernel = np.flip(filterKernel[0])    # flipping the filter kernel befor convolution
    filImg  = np.zeros((N0, N1), dtype=int)

    for i in range(n[0]):
        k = slice(i, i+N0)
        for j in range(n[1]):
            filImg[:, :] += fKernel[i, j]*extImg[k, j:j+N1] 

    return (filImg/filterKernel[1]).astype('uint8')

def gaussFilterKernel(sd=1, k=None):
    
    # radius k computed from the six sigma rule 6*sigma - 1
    # kernel size: w*w = (k + 1 + k)*(k + 1 + k)
    if k is None: k = round(3.0*sd - 0.5) - 1

    _2k     = 2*k
    mat     = np.zeros((_2k+1, _2k+1), dtype=float)
    _2sd2   = 2.0*sd*sd
    _2pisd2 = np.pi*_2sd2    

    for i in range(k+1):
        x = i - k
        for j in range(i, k+1):
            y         = j - k
            a         = _2k - j
            b         = _2k - i
            mat[i, j] = mat[i, a] = mat[j, i] = mat[j, b] = mat[a, i] = mat[a, b] \
                      = mat[b, j] = mat[b, a] = 1.0/(_2pisd2*np.exp((x**2 + y**2)/_2sd2))

    mat = (mat/mat.min()).astype(int)
    return (mat, np.sum(mat))

def histogram(img):
    minVal = img.min()
    maxVal = img.max()
    num, H = np.unique(img, return_counts = True)
    sumH   = H.sum()
    if minVal > _g_min: 
        H   = np.insert(H, 0, np.zeros(minVal))
        num = np.insert(num, 0, np.arange(minVal))
    # if maxVal < _g_max:
    #     H   = np.insert(H, len(num), np.zeros(_g_max - maxVal))
    #     num = np.insert(num, len(num), np.arange(maxVal + 1, _g_max + 1))
    if num.size < maxVal + 1:
        j   = minVal
        for i in range(j, maxVal):
            if i < num[j]: 
                H   = np.insert(H, j, 0)
                num = np.insert(num, j, i)
                if num.size > maxVal: break
            j += 1

    return (H, sumH, minVal, maxVal)


# finding T (intensity threshold defining if a pixel is 
# black (object) or white (background)) where
# overlap between both groups is minimized 
# (largest variance defference between both)
def otsuBinarization(img, dT, fac):
    # building a histogram 
    H, sumH, minVal, maxVal = histogram(img)
    h                       = H/sumH
    uH                      = np.dot(np.arange(H.size), H)
    cI                      = sumbl = uHbl = Smax = 0
    uold                    = minVal
    u                       = T = uold + dT

    while u <= maxVal:
        cI    += np.sum(h[uold:u])
        sumbl += H[uold:u].sum()
        uHbl  += np.dot(np.arange(uold, u), H[uold:u])
        # variance difference calculation
        sig2  =  cI*(1.0 - cI)*(uHbl/sumbl - (uH - uHbl)/(sumH - sumbl))**2

        if sig2 > Smax: # largest possible variance difference
            Smax = sig2
            T    = u

        uold  =  u
        u     += dT

    # boolean: 
    # True: background (white pixel)
    # False: object (black pixel) 
    biImg = img >= T/fac
    
    return (biImg, T)


# 2-pass component-connected labelling
#   0: unlabelled object (black pixel), -1: background (white pixel))
# >=1: labelled object
def findLabel(linked, i):
    if linked[i] < i:
        return findLabel(linked, linked[i])
    return i

def region8(imgRaw):
    N0, N1           = imgRaw.shape
    img              = np.full((N0+2, N1+2), -1, dtype=int) # background (-1) initially
    eN0              = range(1, N0+1)
    eN1              = range(1, N1+1)
    # backgrounds (1) turned to -1 while unlabelled objects stay 0
    img[1:N0+1, eN1] = -1*imgRaw.astype(int)
    iN               = 1
    linked           = np.array(0, dtype=int)  # object equivalence vector
    actNum           = np.array(-1, dtype=int) # pixel number (area) of each object

    # pass 1:
    # 1. assigning temporary labels to each object pixel
    # 2. filling out the object equivalence vector relating each temporary label
    # 3. counting the pixel number (computing the area) of each object
    for i in eN0:
        for j in eN1:
            if img[i, j] == -1: continue # img[i, j] := background -> skip

            # img[i, j] := unlabelled object pixel (first time visited)
            # 
            # left or above (already scanned) neighbours: previous neighbours
            # (traditional scanning order: left -> right, top -> bottom)
            # 1   2   3       
            # 0  ij  
            #       
            neigh = np.array([img[i, j-1], img[i-1, j-1], img[i-1, j], img[i-1, j+1]])
            ind   = np.nonzero(neigh != -1)[0] # only labelled objects (>= 1)
            num   = ind.size                   # no. of scanned, labelled objects

            # all neighbours are backgrounds => new region labelled to img[i, j]
            # as img[i, j] is not connected to previously scanned objects
            if num == 0:
                linked    =  np.append(linked, iN)
                actNum    =  np.append(actNum, 1)
                img[i, j] =  iN
                iN        += 1
            # only one labelled object neighbour => img[i, j] merged to that region
            elif num == 1: 
                img[i, j] =  linked[neigh[ind]]
                actNum[linked[img[i, j]]] += 1
            # more than one object neighbour
            else:
                # get the earliest label number out of all the labelled neighbours
                neighReg  = np.unique(linked[neigh[ind]]) # sorting and uniqing
                img[i, j] = neighReg[0]           # minimum label
                if neighReg.size > 1: 
                    actNum[linked[neighReg[0]]]   += 1 + actNum[linked[neighReg[1:]]].sum()
                    actNum[linked[neighReg[1:]]]  =  0
                    linked[neighReg]              =  neighReg[0]
                else: actNum[linked[neighReg[0]]] += 1

    # pass 2: performed only when there exists at least 1 object
    # 1. squeezing the label numbers (pixel number vector)
    # 2. assigning (correcting) the actual label to each object pixel
    #    with the help of the object equivalence and pixel number vectors 
    if iN > 1:
        num    = np.zeros(linked.size, dtype=int)
        num[1] = 1
        iN     = 2
        for i in range(iN, linked.size):
            j = findLabel(linked, i)
            if i == j:
                num[i]     =  iN
                actNum[iN] =  actNum[i] 
                iN         += 1
            img[img == i] = num[j]
    else:
        print(f"No object found!!!")
        return None, None

    # img with an artificially extended white (-1) border
    img[img != -1] += 1
    return img, np.insert(actNum[:iN], 0, -1)


_g_sqrt2 = math.sqrt(2.0)
# clockwise checking sequence
# 5   6   7       -1,-1  -1,0  -1,1
# 4  ij   0        0,-1   0,0   0,1
# 3   2   1        1,-1   1,0   1,1
# q := previously checked pixel
# p := current pixel

# check if the object is a single-pixel object
# (an object with only one isolated black pixel)
def borderAndSingleCheck(img, obj, peri, i, j):
    # object already having an external border trace
    if len(peri[obj]) > 0:
        # left neighbour is a marked background (0) => skip
        if img[i, j-1] == 0: return None
        # left neighbour is an unmarked background (-1) 
        # => an internal border found (hole inside the object)
        else: num = 1 # never a single-pixel object
    # object having no external border trace yet
    else: # left neighbour is either a marked or unmarked background
        num = ((img[i-1:i+2, j-1:j+2] == -1) |
               (img[i-1:i+2, j-1:j+2] == 0)).sum()

    # marking the border pixel by a negative value
    img[i, j] = -obj

    # all 7 neighbours are white => an isolated single-pixel object
    if num == 7:
        # marking all surrounding background pixels by 0
        img[i-1:i+2:2, j-1:j+2] = img[i, j-1] = img[i, j+1] = 0
        peri[obj].append(1.0)
        return None
    else: img[i, j-1] = 0
    
    return 1


# finding current node index
def nodeIndex(qi, qj, pi, pj):
    if qi < pi:
        if qj < pj:    return (qi, pj), 6, 1.0
        elif qj == pj: return (qi, pj+1), 7, _g_sqrt2
        else:          return (pi, qj), 0, 1.0
    elif qi == pi:
        if qj < pj:    return (qi-1, qj), 5, _g_sqrt2
        else:          return (qi+1, qj), 1, _g_sqrt2
    else:
        if qj < pj:    return (pi, qj), 4, 1.0 
        elif qj == pj: return (qi, pj-1), 3, _g_sqrt2
        else:          return (qi, pj), 2, 1.0

# finding next q around p
def nextNode(qi, qj, pi, pj, k=None):
    if k is None:      return nodeIndex(qi, qj, pi, pj)
    elif k == 5:       return (qi, pj), 6, 1.0
    elif k == 6:       return (qi, pj+1), 7, _g_sqrt2
    elif k == 7:       return (pi, qj), 0, 1.0
    elif k == 4:       return (qi-1, qj), 5, _g_sqrt2
    elif k == 0:       return (qi+1, qj), 1, _g_sqrt2
    elif k == 3:       return (pi, qj), 4, 1.0
    elif k == 2:       return (qi, pj-1), 3, _g_sqrt2
    else:              return (qi, pj), 2, 1.0

def vossTracing(img, obj, qi, qj, pi, pj, k):
    peri             = 0.0
    q0p0             = np.array([qi, qj, pi, pj], dtype=int)
    (qi, qj), k, seg = nextNode(qi, qj, pi, pj, k)  # next q and k around p
    qkpi             = np.array([qi, qj, pi, pj], dtype=int)
    imin             = imax = pi
    jmin             = jmax = pj

    while True:
        # looping over detected object-border pixels  
        # until the checked pixel is not an object border
        while abs(img[qi, qj]) == obj:
            # parameter calculation
            if qi < imin:   imin = qi
            elif qi > imax: imax = qi
            if qj < jmin:   jmin = qj
            elif qj > jmax: jmax = qj
            peri             += seg
            # marking the border pixel by a negative value
            img[qi, qj]      =  -obj

            pi, pj           =  qi, qj   # old q
            # new q <- next q for (q <- old p, p <- old q)
            (qi, qj), k, seg =  nextNode(qkpi[2], qkpi[3], pi, pj)
            qkpi[2:]         =  (pi, pj) # new p <- old q

        # marking the surrounding background pixel by 0
        img[qi, qj]      = 0
        qkpi[:2]         = (qi, qj)
        # stop once the checked point has reached the beginning point
        if np.all(qkpi == q0p0): break
        # next q (neighbour) around p
        (qi, qj), k, seg = nextNode(qi, qj, qkpi[2], qkpi[3], k)

    
    return peri, imin, imax, jmin, jmax


# A := object areas
def perimeterCalc(img, A):
    lnum = A.size
    eN0  = range(1, img.shape[0]-1)
    eN1  = range(1, img.shape[1]-1)
    peri = [[] for i in range(lnum)]
    r    = np.zeros((lnum, 2), dtype=int)
    t    = np.zeros((lnum, 2), dtype=int)


    for i in eN0:
        for j in eN1:
            obj = abs(img[i, j])
            # current pixel i,j is a background pixel
            if obj < 2: continue

            # current pixel i,j is an object pixel
            
            # left neigbour i,j-1 is an object pixel => skip
            if abs(img[i, j-1]) > 1: continue

            # current pixel i,j is not an isolated single-pixel object
            if borderAndSingleCheck(img, obj, peri, i, j) is not None:
                P, imin, imax, jmin, jmax = vossTracing(img, obj, i, j-1, i, j, 4)
                if len(peri[obj]) == 0: 
                    r[obj][0] = imax - imin + 1
                    r[obj][1] = jmax - jmin + 1
                    t[obj][0] = (imax + imin)/2
                    t[obj][1] = (jmax + jmin)/2
                peri[obj].append(P)
    
    return img, A, peri, r, t


# rectangle corner point orientation
# 
#       0                             
#        *- _                
#    ad / ac _* 1            
#      / _ -  |             
#   2 *--___  |            
#       ab  --* 3      
#       
def cornerOrientation(p, index):
    tmp      = index[2]
    ab0, ab1 = p[index[3]] - p[tmp]
    ac0, ac1 = p[index[1]] - p[tmp]
    ad0, ad1 = p[index[0]] - p[tmp]

    # ab x ac
    if (ab0*ac1 - ab1*ac0) > 0:       # CCW
        # ac x ad
        if (ac0*ad1 - ac1*ad0) > 0:   # CCW
            #
            #     0 (3)                     
            #      *- _                     
            #  ad / ac _* 1 (2)           
            #    / _ -  |            
            # 2 *--___  |            
            # (0) ab  --* 3 (1)    
            #
            index[2] = index[1]
            index[1] = index[3]
            index[3] = index[0]
            index[0] = tmp
        # ab x ad 
        elif (ab0*ad1 - ab1*ad0) > 0: # CCW
            #
            #     1 (3)                     
            #      *- _                     
            #  ac / ad _* 0 (2)           
            #    / _ -  |            
            # 2 *--___  |            
            # (0) ab  --* 3 (1)    
            #
            index[2] = index[0]
            index[0] = tmp 
            tmp      = index[1]
            index[1] = index[3]
            index[3] = tmp
        else:
            #
            #     1 (3)                     
            #      *- _                     
            #  ac / ab _* 3 (2)           
            #    / _ -  |            
            # 2 *--___  |            
            # (0) ad  --* 0 (1)    
            #
            index[2] = index[3]
            index[3] = index[1]
            index[1] = index[0]
            index[0] = tmp
    # ab x ad
    elif (ab0*ad1 - ab1*ad0) > 0:     # CCW
        #
        #     0 (3)                     
        #      *- _                     
        #  ad / ab _* 3 (2)           
        #    / _ -  |            
        # 2 *--___  |            
        # (0) ac  --* 1 (1)    
        #
        index[2] = index[3]
        index[3] = index[0]
        index[0] = tmp
    # ac x ad
    elif (ac0*ad1 - ac1*ad0) > 0:     # CCW
        #
        #     3 (3)
        #      *- _
        #  ab / ad _* 0 (2)
        #    / _ -  |
        # 2 *--___  |
        # (0) ac  --* 1 (1)
        #
        index[2] = index[0]
        index[0] = tmp
    else:
        #
        #     3 (3)
        #      *- _
        #  ab / ac _* 1 (2)
        #    / _ -  |
        # 2 *--___  |
        # (0) ad  --* 0 (1)
        #
        index[2] = index[1]
        index[1] = index[0]
        index[0] = tmp
        
    point = np.array([p[index[0]], p[index[1]], p[index[2]], p[index[3]]])
    delp  = np.array([point[1, :] - point[0, :], point[2, :] - point[1, :], 
                      point[3, :] - point[2, :], point[0, :] - point[3, :]])

    return point, delp


def originLocating(p, delp, x):
    
    # 1. detecting a point in a convex quadrilateral
    #    by checking if it stays to the left of all four edges
    # 
    #        3 (d)                          
    #          *- _                
    #         / *x _* 2 (c)           
    #        / _ -  |              
    # (a) 0 *--___  |            
    #             --* 1 (b)      
    #
    # 1.1. checking if point x stays to the left of edge 01 (ab)
    ax0, ax1 = x - p[0]
    # ax x ab
    if (ax0*delp[0, 1] - ax1*delp[0, 0]) > 0:
        return None, None  # x located to the right of edge 01 (ab)
    # 1.2. checking if point x stays to the left of edge 12 (bc)
    bx0, bx1 = x - p[1]
    # bx x bc
    if (bx0*delp[1, 1] - bx1*delp[1, 0]) > 0:
        return None, None  # x located to the right of edge 12 (bc)
    # 1.3. checking if point x stays to the left of edge 23 (cd)
    cx0, cx1 = x - p[2]
    # cx x cd
    if (cx0*delp[2, 1] - cx1*delp[2, 0]) > 0:
        return None, None  # x located to the right of edge 23 (cd)
    # 1.4. checking if point x stays to the left of edge 30 (da)
    dx0, dx1 = x - p[3]
    # dx x da
    if (dx0*delp[3, 1] - dx1*delp[3, 0]) > 0:
        return None, None  # x located to the right of edge 30 (da)

    # 2. locating the actual origin (corner point closest to point x)
    dist = np.array([ax0*ax0 + ax1*ax1, bx0*bx0 + bx1*bx1, cx0*cx0 + cx1*cx1, dx0*dx0 + dx1*dx1])
    mind = np.min(dist)
    minI = np.argmin(dist)

    return mind, minI


def cornerReording(p, minI):
    # 1. resorting the corner points with respect to the found actual origin regarded as point 0
    if minI == 1:   # p[1] := origin
        tmp     = copy.deepcopy(p[0])
        p[0, :] = p[1, :]
        p[1, :] = p[2, :]
        p[2, :] = p[3, :]
        p[3, :] = tmp[:]
    elif minI == 2: # p[2] := origin
        tmp     = copy.deepcopy(p[3])
        p[3, :] = p[1, :]
        p[1, :] = tmp[:]
        tmp     = copy.deepcopy(p[2])
        p[2, :] = p[0, :]
        p[0, :] = tmp[:]
    elif minI == 3: # p[3] := origin
        tmp     = copy.deepcopy(p[3])
        p[3, :] = p[2, :]
        p[2, :] = p[1, :]
        p[1, :] = p[0, :]
        p[0, :] = tmp[:]

    return p


#
#       3 (0,yNum-1)
#        *- _
#       /     * 2 (xNum-1,yNum-1)
#      /      |
#   0 *--___  |
#  (0,0)    --* 1 (xNum-1,0)
#
# CHAPTER 2:
# square system matrix => LU decomposition (solving system of linear equations)
# (x0, y0) = (        0,         0) <origin>
# (x1, y1) = (xN=xNum-1,         0) <x-axis end>
# (x2, y2) = (xN=xNum-1, yN=yNum-1) <upper right corner>
# (x3, y3) = (        0, yN=yNum-1) <y-axis end>
#
#              Homography matrix (nonaffine => h20, h21, h22 != 0)
#     (linear perspective transformation or direct linear transformation (DLT))
#  _   _     _         _     _               _     _    _      _     _     _   _
# |  X  |   |  alpha*x  |   |  h00  h01  h02  |   |  ti  |    |  X/Z  |   |  x  |
# |  Y  | = |  alpha*y  | = |  h10  h11  h12  | x |  tj  | => |  Y/Z  | = |  y  |
# |_ Z _|   |_ alpha   _|   |_ h20  h21  h22 _|   |_  1 _|    |_ Z/Z _|   |_ 1 _|
#  _                                                _  _   _     _   _
# |  -ti0  -tj0  -1    0     0    0  ti0.x0  tj0.x0  || h00 |   | -x0 |
# |    0     0    0  -ti0  -tj0  -1  ti0.y0  tj0.y0  || h01 |   | -y0 |
# |  -ti1  -tj1  -1    0     0    0  ti1.x1  tj1.x1  || h02 |   | -x1 |
# |    0     0    0  -ti1  -tj1  -1  ti1.y1  tj1.y1  || h10 |   | -y1 |
# |  -ti2  -tj2  -1    0     0    0  ti2.x2  tj2.x2  || h11 | = | -x2 |
# |    0     0    0  -ti2  -tj2  -1  ti2.y2  tj2.y2  || h12 |   | -y2 |
# |  -ti3  -tj3  -1    0     0    0  ti3.x3  tj3.x3  || h20 |   | -x3 |
# |_   0     0    0  -ti3  -tj3  -1  ti3.y3  tj3.y3 _||_h21_|   |_-y3_| # 9th constraint added  
#                                                                       # h22 = 1
#                                                                        => square H + non-homogeneous 
# both A and rhs are scaled to reduce the condition number
def fourPointHomography(p, xN, yN):
    t      = np.append(p, [[1], [1], [1], [1]], axis=1)
    zeroAr = [0, 0, 0]
    zeroT  = [0, 0]
    # elements of the scaled system matrix A
    fac    = np.abs(t[:, 0]) + np.abs(t[:, 1]) + 1
    a0     = -t[0, :]/fac[0]
    a2     = -t[1, :]/(fac[1]*(1 + xN) - xN)
    a4     = -t[2, :]/(fac[2]*(1 + xN) - xN)
    a5     = -t[2, :]/(fac[2]*(1 + yN) - yN)
    a7     = -t[3, :]/(fac[3]*(1 + yN) - yN)

    A      = np.array([np.concatenate((             a0,          zeroAr,      zeroT)),
                       np.concatenate((         zeroAr,              a0,      zeroT)),
                       np.concatenate((             a2,          zeroAr, -a2[:2]*xN)),
                       np.concatenate((         zeroAr, -t[1, :]/fac[1],      zeroT)),
                       np.concatenate((             a4,          zeroAr, -a4[:2]*xN)),
                       np.concatenate((         zeroAr,              a5, -a5[:2]*yN)),
                       np.concatenate((-t[3, :]/fac[3],          zeroAr,      zeroT)),
                       np.concatenate((         zeroAr,              a7, -a7[:2]*yN))], dtype=float)
    rhs    = np.array([0, 0, a2[2]*xN, 0, a4[2]*xN, a5[2]*yN, 0, a7[2]*yN], dtype=float)
    x_hat  = nuMet.gaussElimination(A, rhs, True)

    return np.reshape(np.append(x_hat, 1.0), (3, 3))
    


# keypoint detection for calibration
class keypoint():
    def __init__(self, subimg, theta=0, k=2, t=np.zeros(2, dtype=float), 
                 invh1=1, invh2=1, L1=_g_min, L2=_g_max):
        self.subimg = subimg
        self.theta  = theta            # 0
        self.k      = k                # 1
        self.t      = copy.deepcopy(t) # [2, 3]
        self.invh1  = invh1            # 4
        self.invh2  = invh2            # 5
        self.L1     = float(L1)        # 6
        self.L2     = float(L2)        # 7


    def upperBlockQRdecomposition(self, A, b, m, n):
        Rw, Rdi, j                   = nuMet.householderDecomposition(A, m, n)
        Qb                           = nuMet.householderRHStall(Rw, b, m, n)
        Rw[np.tril_indices(m, 0, n)] = 0 # all zero elements on the lower triangular part
        np.fill_diagonal(Rw, Rdi)        # actual diagonal elements of Rw

        return Rw, Qb


    def getData(self):
        return np.array([self.theta, self.k, self.t[0], self.t[1], 
                         self.invh1, self.invh2, self.L1, self.L2])

    
    def updateData(self, Xnew):
        (self.theta, self.k, self.t[0], self.t[1], 
         self.invh1, self.invh2, self.L1, self.L2) = Xnew[:]
        return


    def getCentre(self):
        return self.t


    def updateCentre(self, start):
        self.t = self.t + start
        return


    # transforming points (i, j) on the ellipse in the matrix (pixel) coordinates
    #           to points (x, y) on the unit circle in the cartesian coordinates
    #     (minor
    #      axis) ^ y    (a unit circle)
    #          __|__
    #        /   |   \    x (major axis)
    #  -----|----+----|-------->-----------------------------> j
    #        \   |\   /
    #  (x, y) *--|--* 
    #            |  \
    #            |   \ tij         major axis (longer radius)
    #            |    \          /
    #      X _   |     \        /_
    #          X _      \   O  O   -_ theta (counterclockwise)
    #   minor    | X _   O    /  O   -_
    #   axis     |     X _ \ / h1 O   _
    #  (shorter  |    O h2 X+-----O---------------> positive x
    #   radius)  |    O          O
    #            |     O       O
    #            |       O  * (i, j)
    #            |   (an ellipse)
    #           \/
    #            i                                             unit circle    ellipse                
    #    3.                  2.                       3.          point        pixel
    #  shifting           rotating                  scaling    coordinates   location
    #  _     _     _                        _     _         _     _   _       _    _
    # |   tj  | + |  cos(theta) -sin(theta)  | * |  h1   0   | * |  x  |  =  |   j  |
    # |_ -ti _|   |_ sin(theta)  cos(theta) _|   |_ 0    h2 _|   |_ y _|     |_ -i _|
    #  _             _     _                        _     _           _       _    _
    # |  h1^-1   0    | * |   cos(theta) sin(theta)  | * |    j - tj   |  =  |   x  |
    # |_   0   h2^-1 _|   |_ -sin(theta) cos(theta) _|   |_ -(i - ti) _|     |_  y _|
    #           _                                      _     _        _       _    _
    #          |  -h1^-1*sin(theta)   h1^-1*cos(theta)  | * |  i - ti  |  =  |   x  |
    #          |_ -h2^-1*cos(theta)  -h2^-1*sin(theta) _|   |_ j - tj _|     |_  y _|
    # 
    # - d := point distance from the centre shifted by the radius of the unit circle 
    #        to make d = 0 exactly at the circle edge and thus ellipse 
    #        (negative inside and positive outside the object)
    #                    ___________
    #              d = \/ x^2 + y^2  - 1 
    # 
    # - L~ := normalized intensity (-1 <= L~ <= 1)
    # 
    #                L~       white
    #                 |   ________________ 1         |-  1 if kd > 1 (outside the ellipse)
    #                 |  /                           |  
    #                 | /--- slope = k         L~ = -|- kd if -1 <= kd <= 1 (across the edge)
    #                 |/                             |
    #      -----------+----------------> d           |- -1 if kd < -1 (inside the ellipse)
    #                /|                       
    #        black  / |
    #   -1 ________/  |  L~ = 0 at the circle (ellipse) edge
    #                 |

    def errorFunction(self):
        (h, w)        = self.subimg.shape
        self.vecDim   = h*w
        self.c        = math.cos(self.theta)
        self.s        = math.sin(self.theta)
        self.invh1cos = self.invh1*self.c
        self.invh1sin = self.invh1*self.s
        self.invh2cos = self.invh2*self.c
        self.invh2sin = self.invh2*self.s
        A             = np.array([[-self.invh1sin,  self.invh1cos], 
                                  [-self.invh2cos, -self.invh2sin]], dtype=float)
        self.U        = np.array(np.meshgrid(np.arange(h) - self.t[0],
                                             np.arange(w) - self.t[1], 
                                             indexing='ij')).reshape((2, self.vecDim))
        self.X        = A@self.U
        self.dc       = np.sqrt(np.einsum('ij,ij->j', self.X, self.X))
        self.d        = self.dc - 1
        kd            = self.k*self.d
        kd[kd >  1]   = 1
        kd[kd < -1]   = -1
        self.Ltilde   = kd
        # 
        #                                            (L2 - L1)
        # acutal intensity L(x, y) = L1 + (L~ + 1) * ---------
        #                                                2
        # 
        # f(theta, k, ti, tj, invh1, invh2, L1, L2) = fx = L(x, y) - I(i, j)   
        #  
        # ||    ||2
        # || fx ||   = fx.fx 
        # ||    ||2         
        # 
        fx            = (self.L1 + (kd + 1)*
                         (self.L2 - self.L1)/2) - self.subimg.reshape(self.vecDim)

        return fx, np.dot(fx, fx)


    #                                  1          
    # acutal intensity L(x, y) = L1 + --- * (L~ + 1) * (L2 - L1)
    #                                  2            
    #                       #0  #1  #2  #3   #4     #5    #6  #7
    # error function := f(theta, k, ti, tj, h1^-1, h2^-1, L1, L2) = f(var) = fx = L(x, y) - I(i, j)
    #                                _            _           _                   _
    # df(var)   dL(x, y)    dL1   1 |            |  dL2    dL1 |              dL~  |
    # ------- = -------- = ---- + -*| (L~ + 1) * | ---- - ---- | + (L2 - L1)*----  |
    #  dvar       dvar     dvar   2 |_           |_dvar   dvar_|             dvar _|
    #                                         
    #      df(var)           dL(x, y)       (L2 - L1)  dL~ 
    #  --------------- = --------------- =  ---------*---- 
    #  d(var\{L1, L2})   d(var\{L1, L2})        2     dvar 
    # 
    # - normalized intensity:    
    #               dL~      dd       dk
    #   L~ = kd => ---- = k*---- + d*----
    #              dvar     dvar     dvar
    # 
    # - shifted location distance from the circle centre: 
    #                                                   _                 _
    #          ___________                  dd     1   |     dx       dy   |
    #    d = \/ x^2 + y^2  - 1 = dc - 1 => ---- = -- * |  x*---- + y*----  |
    #                                      dvar   dc   |_   dvar     dvar _|
    #                               _               _               _    _                 _          _
    #      df(var)       (L2 - L1) |    dd       dk  |   (L2 - L1) |  k |     dx       dy   |      dk  |
    #  --------------- = ---------*| k*---- + d*---- | = ---------*| --*|  x*---- + y*----  | + d*---- |
    #  d(var\{L1, L2})       2     |_  dvar     dvar_|       2     |_dc |_   dvar     dvar _|     dvar_|
    # 
    # - coordinate transformation from the ellipse matrix coordinates (i, j) to the circle cartesian ones (x, y):
    #           _                                      _     _        _       _    _
    #          |  -h1^-1*sin(theta)   h1^-1*cos(theta)  | * |  i - ti  |  =  |   x  |
    #          |_ -h2^-1*cos(theta)  -h2^-1*sin(theta) _|   |_ j - tj _|     |_  y _|
    # 
    #    x = [-(i - ti)*sin(theta) + (j - tj)*cos(theta)]*h1^-1
    #               _                                                                                            _
    #  dx          |                      dtheta               dti                       dtheta               dtj |
    # ---- = h1^-1*| -(i - ti)*cos(theta)*------ + sin(theta)*---- - (j - tj)*sin(theta)*------ - cos(theta)*---- |
    # dvar         |_                      dvar               dvar                        dvar               dvar_|
    #
    #                                                           dh1^-1
    #            + [-(i - ti)*sin(theta) + (j - tj)*cos(theta)]*------
    #                                                            dvar
    #
    #    y = [-(i - ti)*cos(theta) - (j - tj)*sin(theta)]*h2^-1
    #               _                                                                                           _
    #  dy          |                     dtheta               dti                       dtheta               dtj |
    # ---- = h2^-1*| (i - ti)*sin(theta)*------ + cos(theta)*---- - (j - tj)*cos(theta)*------ + sin(theta)*---- |
    # dvar         |_                     dvar               dvar                        dvar               dvar_|
    #
    #                                                           dh2^-1
    #            + [-(i - ti)*cos(theta) - (j - tj)*sin(theta)]*------
    #                                                            dvar
    # 
    # Partial derivatives (gradient) of f with respect to all vars:
    # 
    #      df(var)   (L2 - L1)   
    # (1)  ------- = ---------*d 
    #        dk          2       
    #                            _    _                 _          _                 _                _
    #      df(var)    (L2 - L1) |  k |     dx       dy   |      dk  |   (L1 - L2)*k |   dx       dy    |
    # (0)  -------  = ---------*| --*|  x*---- + y*----  | + d*---- | = -----------*|  ----*x + ----*y |  
    #       dtheta        2     |_dc |_   dvar     dvar _|     dvar_|      2*dc     |_ dvar     dvar  _| 
    #                             _        _                                          _   _
    #                            |        |                                            |   |
    #                            |  h1^-1*| -(i - ti)*cos(theta) - (j - tj)*sin(theta) |*x |
    #                (L2 - L1)*k |        |_                                          _|   |
    #              = -----------*|          _                                         _    |
    #                   2*dc     |         |                                           |   |
    #                            | + h2^-1*| (i - ti)*sin(theta) - (j - tj)*cos(theta) |*y |
    #                            |_        |_                                         _|  _|
    #                (L2 - L1)*k  _                 _     _                                      _     _   _
    #              = -----------*|_ i - ti   j - tj _| * |  -h1^-1*cos(theta)   h2^-1*sin(theta)  | * |  x  |
    #                   2*dc                             |_ -h1^-1*sin(theta)  -h2^-1*cos(theta) _|   |_ y _|
    #                               _               _                 _                                        _
    #      df(var)   (L2 - L1)*k   |    dx       dy  |   (L1 - L2)*k |                                          | 
    # (2)  ------- = ----------- * | x*---- + y*---- | = -----------*|  h1^-1*sin(theta)*x + h2^-1*cos(theta)*y |  
    #        dti        2*dc       |_  dvar     dvar_|      2*dc     |_                                        _|
    #                (L2 - L1)*k    _                                    _     _   _
    #              = ----------- * |                                      | * |  x  |
    #                   2*dc       |_ h1^-1*sin(theta)  h2^-1*cos(theta) _|   |_ y _|
    #                             _               _                 _                                        _
    #      df(var)   (L2 - L1)*k |    dx       dy  |   (L1 - L2)*k |                                          | 
    # (3)  ------- = -----------*| x*---- + y*---- | = -----------*| -h1^-1*cos(theta)*x + h2^-1*sin(theta)*y |  
    #        dtj        2*dc     |_  dvar     dvar_|      2*dc     |_                                        _|
    #                (L2 - L1)*k    _                                     _     _   _
    #              = ----------- * |  -h1^-1*cos(theta)  h2^-1*sin(theta)  | * |  x  |
    #                   2*dc       |_                                     _|   |_ y _|
    #                             _               _                 _                                        _
    #      df(var)   (L2 - L1)*k |    dx       dy  |   (L1 - L2)*k |                                          | 
    # (4)  ------- = -----------*| x*---- + y*---- | = -----------*|-(i - ti)*sin(theta) + (j - tj)*cos(theta)|*x
    #       h1^-1       2*dc     |_  dvar     dvar_|      2*dc     |_                                        _|
    #                (L2 - L1)*k  _                 _     _             _     
    #              = -----------*|_ i - ti   j - tj _| * |  -sin(theta)  | * x
    #                   2*dc                             |_  cos(theta) _|   
    #                             _               _                 _                                        _
    #      df(var)   (L2 - L1)*k |    dx       dy  |   (L1 - L2)*k |                                          | 
    # (5)  ------- = -----------*| x*---- + y*---- | = -----------*|-(i - ti)*cos(theta) - (j - tj)*sin(theta)|*y
    #       h2^-1       2*dc     |_  dvar     dvar_|      2*dc     |_                                        _|
    #                (L2 - L1)*k  _                 _     _             _     
    #              = -----------*|_ i - ti   j - tj _| * |  -cos(theta)  |  
    #                   2*dc                             |_ -sin(theta) _| * y
    # 
    #      df(var)   L~ + 1
    # (7)  ------- = ------
    #        dL2        2
    # 
    #      df(var)       L~ + 1       df(var)
    # (6)  ------- = 1 - ------ = 1 - -------
    #        dL1           2            dL2
    # 
    #
    def dFx(self, fx):
        dfx       = np.zeros((self.vecDim, 8), dtype=float)
        Lratio    = (self.L2 - self.L1)/2.0
        
        dfx[:, 1] = Lratio*self.d                            # 1. dfx/dk
        
        Lratio    = Lratio*self.k/self.dc
        A         = np.array([[-self.invh1cos,  self.invh2sin], 
                              [-self.invh1sin, -self.invh2cos]], dtype=float)
        b         = A@self.X
        dfx[:, 0] =  Lratio*np.einsum('ij,ij->j', self.U, b) # 0. dfx/dtheta
        dfx[:, 2] = -Lratio*b[1, :]                          # 2. dfx/dti
        dfx[:, 3] =  Lratio*b[0, :]                          # 3. dfx/dtj
        A         = np.array([[-self.s, -self.c], 
                              [ self.c, -self.s]], dtype=float)
        b         = self.U.T@A
        dfx[:, 4] =  Lratio*b[:, 0]*self.X[0, :]             # 4. dfx/dh1^-1
        dfx[:, 5] =  Lratio*b[:, 1]*self.X[1, :]             # 5. dfx/dh2^-1
        
        dfx[:, 7] = (self.Ltilde + 1)/2                      # 7. dfx/L2
        dfx[:, 6] = 1 - dfx[:, 7]                            # 6. dfx/L
        
        # zero gradient on all black and white pixels outside the edge zone
        dfx[np.abs(self.Ltilde) > 0.9999999, :] = 0         

        fx_dfx    = fx@dfx

        return dfx, np.dot(fx_dfx, fx_dfx)


# blacking a concentric circle from the grey image row by row 
# on each of which the process starts from the centre outwards until reaching
# the darkest pixels (minimum black) on both left and right sides
def fillGrey(img, Ind0, val):
    iInd, num = np.unique(Ind0, return_counts=True)
    a         = num.size
    sortImg   = [[] for i in range(a)]

    j = 0
    for i in range(a):
        sortImg[i].extend(img[j:j+num[i]])
        j   += num[i]
        cen =  int(num[i]/2)
        if sortImg[i][cen] <= val: continue

        k = l = cen - 1
        while k > 0:
            k1 = k - 1
            if sortImg[i][k1] < sortImg[i][k]:   l = k1
            elif sortImg[i][k1] > sortImg[i][k]: break
            k  = k1
        if l != cen - 1: sortImg[i][l:cen+1] = [val for k in range(l, cen+1)]

        k   = l = cen + 1
        lim = num[i] - 1
        while k < lim:
            k1 = k + 1
            if sortImg[i][k1] < sortImg[i][k]:   l = k1
            elif sortImg[i][k1] > sortImg[i][k]: break
            k  = k1
        if l != cen + 1: sortImg[i][cen+1:l+1] = [val for k in range(cen, l)]

    return [element for sublist in sortImg for element in sublist]

# blacking a concentric circle from the binary image row by row
def fillBlack(img, val):
    
    # permanently removing background pixels row by row
    for i in range(img.shape[0]):
        Ind = np.nonzero(img[i, :] == -val)[0] # -val = border pixel
        # from the left most to right most border pixels across all object
        # and background pixels of the donut object
        if Ind.size > 0: img[i, Ind[0]:Ind[-1]+1] = val 
        
    return img


def ellipseParams(img, greyImg, label, r, t, key=None):
    if key is None:
        w     = (5*r.sum()/16).astype(int) # scaled average radius
        start = t - w                      # lower bounds from the centre
        stop  = t + w + 1                  # upper bounds from the centre
        if np.any(start < 0) or np.any(stop > img.shape):
            print(f"keypoint {t[0], t[1]} staying too close to the image border")
            return None
        obj   = greyImg[start[0]:stop[0], start[1]:stop[1]]
        
        # statistical initialization of h1, h2 and theta of object i
        # where the object assumed to have an elliptical shape with h1 being
        # the length of the major radius, h2 being the length of the minor radius
        # and theta being the inclination angle measured counterclockwisely  
        # from the image positive x axis (+----->) to the ellipse major axis):
        #  
        # 1. solving for the eigenvalues of the covariance matrix of 
        #    the column and row indices (x and y coordinates of each object pixel)
        # |  _              _     _            _  |
        # | |   varX  varXY  | - |  lamb    0   | | = 0
        # | |_ varXY   varY _|   |_  0    lamb _| |
        # |                                       |   
        #                                  ____________________________________________
        #                                 /             2
        # lamb1 (max) = (varX + varY) + \/ (varX + varY)  - 4*(varX*varY - varXY*varXY)
        #               ---------------------------------------------------------------
        #                                             2
        #                                  ____________________________________________
        #                                 /             2
        # lamb1 (min) = (varX + varY) - \/ (varX + varY)  - 4*(varX*varY - varXY*varXY)
        #               ---------------------------------------------------------------
        #                                             2
        # 2. calculating h1, h2 and theta based on the probability theory 
        #    (chi-squared distribution) with predefined confidence percentage that  
        #    any labelled pixel would fall inside this ellipse e.g.
        #   2
        # X  (5.991) = 0.95 (95% confidence)
        #  2     
        #
        #   2
        # X  (4.605) = 0.90 (90% confidence)
        #  2     
        #                                              ____________
        # h1 (length of the major (longer) radius) = \/ 5.991*lamb1 
        #                                               ____________
        # h2 (length of the minor (shorter) radius) = \/ 5.991*lamb2
        # 
        # theta = arctan2(varX - h1, varXY) 
        # (the computed angle value ranges from -pi to pi
        #  covering all four quadrants not like the arctan function
        #  where the value ranges from -pi/2 to pi/2)
        #  
        Xi    = 5.991
        Ind   = np.nonzero(np.abs(img[start[0]:stop[0], start[1]:stop[1]]) == label)
        x     = Ind[1]        # column indices of all pixels in object i  +---->
                              # (starting from 0 not start)               |
        y     = Ind[0]        # row indices of all pixels in object i     |
        area  = x.size        # == A[i]
        muX   = x.sum()/area  # average column index (x bar)
        muY   = y.sum()/area  # average row index (y bar)
        varX  = np.dot(x, x)/area - muX*muX
        varY  = np.dot(y, y)/area - muY*muY
        varXY = np.dot(x, y)/area - muX*muY
        root  = math.sqrt((varX + varY)**2.0 - 4.0*(varX*varY - varXY*varXY))
        invh1 = (varX + varY + root)/2.0
        invh2 = 1.0/math.sqrt(Xi*(varX + varY - root)/2.0)
        theta = math.atan2(varX - invh1, varXY)
        invh1 = 1.0/math.sqrt(Xi*invh1)
        # finding the real location of keypoint (cx, cy)
        # data of the considered subimage around object i
        cen   = np.array([muY, muX])
        k     = 2.0
        key   = keypoint(obj, theta, k, cen, invh1, invh2, obj.min(), obj.max())
    else:
        theta, k, cen, invh1, invh2 = key.theta, key.k, key.t, key.invh1, key.invh2
        w                           = copy.deepcopy(r)
        start                       = copy.deepcopy(r)


    itera = nuMet.levenbergMarquardt(key, mu=1e-5, zmax=40, tol=1e-5)
    wlim  = 1.5*w
    oriW  = w
    i     = 1
    while itera == 40:
        w     =  oriW + i
        start =  t - w        # lower bounds from the centre
        stop  =  t + w + 1    # upper bounds from the centre
        if np.any(w > wlim) or np.any(start < 0) or np.any(stop > greyImg.shape):
            print(f"problems precisely locating keypoint {t[0], t[1]}")
            return None
        obj   =  greyImg[start[0]:stop[0], start[1]:stop[1]]
        cen   =  cen + 1
        key   =  keypoint(obj, theta, k, cen, invh1, invh2, obj.min(), obj.max())
        itera =  nuMet.levenbergMarquardt(key, mu=1e-5, zmax=40, tol=1e-5)
        if i < 0: i = -(i - 1)
        else:     i = -(i + 1)

    key.updateCentre(start)

    return key


def drawKeypoint(img, cen, a, b, theta, color):
    center = (round(cen[1]), round(cen[0]))
    cv2.ellipse(img, center, (int(a), int(b)),
                180 - theta*180/np.pi, 0, 360, color, 1)
    cv2.circle(img, center, radius=1, color=color, thickness=2)

    return


def drawCoordinates(img, cen, cx, cy):
    color  = (0, 0, 255) 
    cv2.putText(img, f"({cx}, {cy})", (round(cen[1])-30, round(cen[0])-40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return

class projectingCenter():
    def __init__(self, keyIn, keyOut, rRat):
        self.rRat2 = rRat*rRat
        self.t     = np.array([copy.deepcopy(keyIn.t), copy.deepcopy(keyOut.t)], 
                              dtype=float)
        self.c2    = np.array([keyIn.c*keyIn.c, keyOut.c*keyOut.c], dtype=float)
        self.s2    = np.array([keyIn.s*keyIn.s, keyOut.s*keyOut.s], dtype=float)
        self.cs    = np.array([keyIn.c*keyIn.s, keyOut.c*keyOut.s], dtype=float)
        self.ain   = 1.0/keyIn.invh1
        self.bin   = 1.0/keyIn.invh2
        self.aout  = 1.0/keyOut.invh1
        self.bout  = 1.0/keyOut.invh2
        self.a2    = np.array([self.ain*self.ain, self.aout*self.aout], dtype=float)
        self.b2    = np.array([self.bin*self.bin, self.bout*self.bout], dtype=float)
        # determination of the initial concentric ratio based on the picture
        self.rinit = self.errorFunction() + self.rRat2 
        self.updateData((self.aout - self.ain*rRat)/(1 + rRat))

    
    def getRinit(self):
        return self.rinit


    def getData(self):
        return self.dh

    
    def updateData(self, dhnew):
        self.dh    =  dhnew

        # inner conic
        self.a2[0] =  self.ain + dhnew
        self.a2[0] *= self.a2[0]
        self.b2[0] =  self.bin + dhnew
        self.b2[0] *= self.b2[0]

        # outer conic
        self.a2[1] =  self.aout - dhnew
        self.a2[1] *= self.a2[1]
        self.b2[1] =  self.bout - dhnew
        self.b2[1] *= self.b2[1]
        
        return


    def updateCentre(self):
        try:
            val, vec = eigs(self.A, k=1, sigma=self.val)
        except Exception as e:
            print(f"problem locating the mutual centre of a concentric circle")
            return None
        cen = self.invQin@vec.view(dtype=float)[:, 0]
        
        return np.array([-cen[1]/cen[2], cen[0]/cen[2]])


    def conicMat(self, inOut):
        mat        = np.zeros((3, 3), dtype=float)
        a2, b2     = self.a2[inOut], self.b2[inOut]
        c2, s2, cs = self.c2[inOut], self.s2[inOut], self.cs[inOut]
        t          = self.t[inOut, :]
        mat[0, 0]  = a2*s2 + b2*c2                                # A
        mat[0, 1]  = mat[1, 0] = (b2 - a2)*cs                     # B/2
        mat[1, 1]  = a2*c2 + b2*s2                                # C
        mat[0, 2]  = mat[2, 0] = -mat[0, 0]*t[1] + mat[0, 1]*t[0] # D/2
        mat[1, 2]  = mat[2, 1] = -mat[0, 1]*t[1] + mat[1, 1]*t[0] # E/2
        mat[2, 2]  = mat[0, 0]*t[1]*t[1] - 2.0*mat[0, 1]*t[1]*t[0] + \
                     mat[1, 1]*t[0]*t[0] - a2*b2                  # F

        return mat

    def errorFunction(self):
        qIn, qOut   = self.conicMat(0), self.conicMat(1)
        self.invQin = np.linalg.inv(qIn)
        self.A      = qOut@self.invQin
        val         = np.linalg.eigvals(self.A)

        dist0       = abs(val[0] - val[1])
        dist1       = abs(val[1] - val[2])
        dist2       = abs(val[2] - val[0])

        if dist0 < dist1:
            if dist0 < dist2: # val[0] ~ val[1]
                Rr2      = 2*val[2]/(val[0] + val[1])
                self.val = val[2]
            else:             # val[2] ~ val[0]
                Rr2      = 2*val[1]/(val[0] + val[2])
                self.val = val[1]
        elif dist1 < dist2:   # val[1] ~ val[2]
            Rr2      = 2*val[0]/(val[1] + val[2])
            self.val = val[0]
        else:                 # val[2] ~ val[0]
            Rr2      = 2*val[1]/(val[0] + val[2])
            self.val = val[1]
        
        return Rr2 - self.rRat2


    def dFx(self, fx):
        self.updateData(self.dh + _g_h)
        dfx = (self.errorFunction() - fx)/_g_h # numerical differentiation

        return dfx


def projectedCenter(rgb, keyIn, keyOut, rRat, color):
    cen  = None
    tole = 1e-10
    while cen is None:
        proCen =  projectingCenter(keyIn, keyOut, rRat)
        z      =  nuMet.dampedNr(proCen, tol=tole, zmax=40, lambmin=0.001)
        cen    =  proCen.updateCentre()
        tole   *= 0.05

    drawKeypoint(rgb, cen, math.sqrt(proCen.a2[0]), 
                 math.sqrt(proCen.b2[0]), keyIn.theta, color)
    drawKeypoint(rgb, cen, math.sqrt(proCen.a2[1]), 
                 math.sqrt(proCen.b2[1]), keyOut.theta, color)

    return cen, proCen.getRinit()

def concentricCenter(img, greyImg, rRat, label, r, t, rgb, color):
    # removing the centre hole 
    # (significant for subsequent keypoint identification processes)
    # 1. permanently removing the centre hole of the binary image (black out)
    inner                 = ((r+4)/2).astype(int)
    start                 = t - inner
    stop                  = t + inner + 1         
    # only border and background pixels within the circle
    img[start[0]:stop[0], 
        start[1]:stop[1]] = fillBlack(img[start[0]:stop[0], start[1]:stop[1]], label)
    # 2. removing the centre hole of the grey image
    # local indices (only object pixels seen in the binary image)
    Ind                   = np.nonzero(img[start[0]:stop[0], start[1]:stop[1]] > 0)
    # global indices (only object pixels seen in the binary image)
    Ind                   = np.array([Ind[0] + start[0], Ind[1] + start[1]])
    # all grey pixels in such a detected black ellipse (binary object pixels)
    mat                   = greyImg[Ind[0, :], Ind[1, :]]                              # ellipse
    blackThres            = mat.min() # mininum black value (darkest grey/black)
    # a copy of all grey pixels in a rectangle containing such a black ellipse
    # (binary object pixels)
    innerGrey             = copy.deepcopy(greyImg[start[0]:stop[0], start[1]:stop[1]]) # rectangle
    
    # 2.1. permanently blacking out all centre-hole (brightest down to the darkest pixels)
    #      => big outer black elliptical object
    greyImg[Ind[0, :], 
            Ind[1, :]]    = fillGrey(mat, Ind[0] - Ind[0].min(), blackThres)
    
    # 2.2. permanently blacking out all peripheral pixels around the big outer black 
    #      elliptical object => small inner white elliptical object
    innerGrey[greyImg[start[0]:stop[0], start[1]:stop[1]] > blackThres] = blackThres
    mat                   = 4*inner + 1
    # expanded subimage filled with the mininum black value
    mat                   = np.ones((mat[0], mat[1]), dtype='uint8')*blackThres
    # original subimage
    obj                   = mat[inner[0]:-inner[0], inner[1]:-inner[1]]   
    # both expanded subimage (mat) and original subimage (obj) also updated 
    obj[:, :]             = innerGrey[:, :] 

    # 2.1. extracting all parameters of the big outer black elliptical object
    key     = ellipseParams(img, greyImg, label, r, t, None)
    if key is None: return None, None

    # 2.2. extracting all parameters of the small inner white elliptical object
    key1    = keypoint(obj, key.theta, -key.k, key.getCentre() - start, 
                       rRat*key.invh1, rRat*key.invh2, blackThres, obj.max())
    key1    = ellipseParams(None, mat, None, inner, t - start + inner, key1)
    if key1 is None: return None, None
    key1.updateCentre(t - 2*inner)
    
    return projectedCenter(rgb, key1, key, rRat, color)


# theoretical maximum possible inverse compactness of any shape is 1/(4*pi) 
# => perfect circle
_g_cmin = 0.85/(4.0*np.pi)      # circular compactness lower limit
_g_cmax = 1.15/(4.0*np.pi)      # circular compactness upper limit

# locating keypoints
def keypointDetermination(rRat_cor, rRat, imgRaw, img, A, P, r, t, camNum, xNum, yNum, keyMem):  
    lnum            = A.size
    rmin            = 6         # noise threshold (minimum keypoint diameter, pixel number)
    corner          = np.zeros(4, dtype=int)  # coordinate marking locations
    ratio           = np.zeros(lnum, dtype=float)
    keyPoint        = np.zeros((lnum, 2), dtype=float)
    xN, yN          = xNum - 1, yNum - 1
    # grey image with its border artificially expanded 
    # by 1 pixel on each side to have a matching size with 
    # that of img (label profile)
    (im, jm)        = img.shape
    greyImg         = np.zeros(img.shape, dtype='uint8')
    greyImg[1:im-1, 
            1:jm-1] = imgRaw[:, :]
    # for the second keypoint locating of all four corner keypoints
    imgCopy         = copy.deepcopy(img)
    greyImgCopy     = copy.deepcopy(greyImg)

    # RGB image
    rgb             = np.zeros((im, jm, 3), dtype='uint8')
    rgb[:, :, 0]    = greyImg[:, :]
    rgb[:, :, 1]    = greyImg[:, :]
    rgb[:, :, 2]    = greyImg[:, :]

    # noise eliminination (removing all untargeted objects)
    A[0:2] = j = 0                                  # marked as an untargeted object
    ratMax = -1
    ratMin = 1000
    for i in range(2, lnum):
        # noise elimination
        # 1. too small object
        # 2. object with holes greater than one (> 1)
        if np.any(r[i] < rmin) or (len(P[i]) > 2):
            A[i] = 0                                # marked as an untargeted object
            continue

        # solid object (no hole)
        if len(P[i]) == 1:
            # circular compactness calculation for solid objects
            Ci = A[i]/(P[i][0]*P[i][0])
            # outside a specified tolerance zone
            # noise elimination
            # 1. non-circular solid object
            if (Ci < _g_cmin) or (Ci > _g_cmax): A[i] = 0 # marked as an untargeted object
        # hollow object (1 hole)
        else:
            # circular compactness calculation for hollow objects
            Ci = A[i]/(P[i][0]*P[i][0] - P[i][1]*P[i][1])
            # within a specified tolerance zone (donut-shaped object)
            if (Ci > _g_cmin) and (Ci < _g_cmax):
                cen, ratAvg    =  concentricCenter(img, greyImg, rRat, i, r[i], 
                                                   t[i], rgb, (0, 0, 255))
                if cen is None: continue
                A[i]           =  0
                ratio[i]       =  ratAvg
                keyPoint[i, :] =  cen[:]
                if ratio[i] < ratMin: ratMin = ratio[i]
                if ratio[i] > ratMax: ratMax = ratio[i]
                j              += 1

            # noise elimination
            # 1. non-circular hollow object
            else: A[i] = 0  # marked as an untargeted object


    if j < xNum*yNum:
        print(f"{j} concentric keypoints found (< {xNum*yNum})")
        return None

    IndR   = ratio.argsort()

    # identifying the corner and inner keypoints
    maxW   = 0.55
    ratAvg = (1 - maxW)*ratMin + maxW*ratMax # weighted sum (average)
    k      = 0
    # from max ratio down to the fifth largest ratio
    for i in IndR[::-1][:5]:
        if ratio[i] > ratAvg: 
            # recomputing the centre of the corner keypoints using the correct radius ratio
            cen, ratMin    =  concentricCenter(imgCopy, greyImgCopy, rRat_cor, i, 
                                               r[i], t[i], rgb, (255, 0, 0))
            if cen is None: break
            
            if k == 4:
                k += 1
                break

            corner[k]      =  i
            keyPoint[i, :] =  cen[:]
            k              += 1

    if k != 4:
        print(f"{k} corner marking points found (!= 4)")
        return None

    # ordering all four corner points counterclockwisely with one of them assumed an origin
    keyCorn, delp = cornerOrientation(keyPoint, corner)

    # finding the smallest dot (used to locate the acutal origin) within the
    # interest quadrilateral zone defined out of the four corners determined above
    Ind = np.argsort(A)
    # looping from the smallest object upwards
    k   = -1
    for i in Ind[A[Ind] != 0]:
        # only a solid circular object is considered
        # point t[i] staying outside the interest zone => skip
        mind, minI =  originLocating(keyCorn, delp, t[i])
        if mind is None: continue

        # small dot closest to the origin
        if k == -1 or mind < closeDist: closeDist, k = mind, minI

    if k == -1:
        print(f"the small dot not found inside the interest quadrilateral zone")
        return None
    # rearranging the corner points
    elif k > 0: cornerReording(keyCorn, k)
    
    # homography matrix built out of the correctly-arranged four concentric corner points
    #  _   _     _         _     _               _     _    _      _     _     _   _
    # |  X  |   |  alpha*x  |   |  h00  h01  h02  |   |  ti  |    |  X/Z  |   |  x  |
    # |  Y  | = |  alpha*y  | = |  h10  h11  h12  | x |  tj  | => |  Y/Z  | = |  y  |
    # |_ Z _|   |_ alpha   _|   |_ h20  h21  h22 _|   |_  1 _|    |_ Z/Z _|   |_ 1 _|
    H = fourPointHomography(keyCorn, xN, yN)
    j = 0
    for i in IndR[ratio[IndR] != 0]:
        cen      = H@np.append(keyPoint[i], 1.0)
        # cx/alpha, cy/alpha
        (cx, cy) = np.around(cen[:2]/cen[2]).astype(int)
        if (cx < 0) or (cx > xN) or (cy < 0) or (cy > yN): continue

        keyMem[cx, cy, :] =  keyPoint[i, :]
        j                 += 1
        # drawing the numbered coordinates of the inner concentric keypoints
        drawCoordinates(rgb, keyMem[cx, cy, :], cx, cy)

    showImg(rgb, camNum)

    if j != xNum*yNum:
        print(f"{j} keyponits found (!= {xNum*yNum})")
        return None

    return keyMem


def showImg(img, camNum):
    global Imgs
    im = cv2.resize(img, (800, 600))
    while len(Imgs) <= camNum:
        Imgs.append(None)
    Imgs[camNum] = im
    # Display the image
    #cv2.imshow('Image', Imgs[camNum])
    #cv2.waitKey(1)

def rgbReg(img):
    # RGB image
    (im, jm)    = img.shape
    rgb         = np.zeros((im, jm, 3), dtype='uint8')
    # Background pixels to be pure white
    ind         = img == -1
    rgb[ind, 0] = rgb[ind, 1] = rgb[ind, 2] = _g_max
    # Object pixels gradually shaded from blue to white
    num         = int(img.max()/_g_max) + 1
    end         = np.min((num, 3))
    fac         = 0
    for i in range(end):
        ind         =  img > fac
        rgb[ind, i] =  np.clip(img[ind] - fac, None, _g_max)
        fac         += _g_max

    return rgb

def findKeypoints(img, rRat_cor, rRat, xNum, yNum,  
                  keyMem, j, sd = 2.0, biFac0 = 3.5):
    if rRat_cor < 2:
        print(f"Too small radius ratio of the corner cencentric keypoints: "
              f"{rRat_cor} < 2")
        return None, None
    elif rRat < 2:
        print(f"Too small radius ratio of the inner cencentric keypoints: "
              f"{rRat} < 2")
        return None, None
    elif rRat_cor < rRat:
        print(f"The radius ratio of the corner cencentric keypoints is {rRat_cor} "
              f"smaller than that of the inner cencentric keypoints of {rRat}")
        return None, None

    blurImg  =  greyScale(img)        # working in the grayscale
    sd       -= 1
    loc      =  None
    biFac    =  biFac0                # initial binarization threshold
    biFacLim =  biFac0*1.9            # binarization threshold upper limit
    itera    =  0

    while loc is None:
        sd       += 1                                  # increasing SD to blur the picture more (more noise reduction)
        blurrFil =  gaussFilterKernel(sd)              # noise reduction using the gaussian blur filter
        blurImg  =  imgConvolution(blurImg, blurrFil)  # blurring
        biFac    =  biFac0
        inc      =  -0.5

        while loc is None and biFac < biFacLim:        # until all keypoints (loc) found or the upper limit reached
            itera             += 1
            print(f"\n{itera}. SD = {sd}, binarization threshold = {biFac}")
            biImg, T          =  otsuBinarization(blurImg, 1, biFac) # Otsu binarization
            reg, S            =  region8(biImg)         # region segmentation (labelling)
            if reg is None:                             # no region found
                biFac = biFac0 - 1.0                    # reducing the threshold (too bright binary picture)
                continue
            rgb               =  rgbReg(reg)
            #showImg(rgb)
            label, A, P, r, t =  perimeterCalc(reg, S)  # calculating the perimeter and area of all regions
            # determining the subpixel 2D location of all keypoints on the picture
            loc               =  keypointDetermination(rRat_cor, rRat, blurImg, label,  
                                                       A, P, r, t, j, xNum, yNum, keyMem)
            if loc is None:                             # not all keypoinys found
                biFac += inc                            # binarization threshold adjustment (either greater or smaller)
                inc   *= -2   

    return sd, biFac

if __name__ == "__main__":
    loc       = np.zeros((10, 6, 2), dtype=float)
    img       = cv2.imread("./cameraCalibration/img/Asset1.jpg")
    sd, biFac = findKeypoints(img, 7.0/2.0, 6.0/2.0, 10, 6, loc)
    sd, biFac = findKeypoints(img, 7.0/2.0, 6.0/2.0, 10, 6, loc, sd, biFac)

# %%
