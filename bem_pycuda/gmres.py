from numpy  import zeros, array, dot, arange, exp, sqrt, random, transpose, sum, savetxt, save
from numpy  import float64 as REAL
from numpy.linalg           import norm
from scipy.linalg           import lu_solve, solve
from scipy.sparse.linalg    import gmres 
import time
from matrixfree import gmres_dot as gmres_dot
from pycuda.tools import DeviceMemoryPool

def GeneratePlaneRotation(dx, dy, cs, sn):

    if dy==0:
        cs = 1. 
        sn = 0. 
    elif (abs(dy)>abs(dx)):
        temp = dx/dy
        sn = 1/sqrt(1+temp*temp)
        cs = temp*sn
    else:
        temp = dy/dx
        cs = 1/sqrt(1+temp*temp)
        sn = temp*cs

    return cs, sn

def ApplyPlaneRotation(dx, dy, cs, sn):
    temp = cs*dx + sn*dy
    dy = -sn*dx + cs*dy
    dx = temp

    return dx, dy

def PlaneRotation (H, cs, sn, s, i, R):
    for k in range(i):
        H[k,i], H[k+1,i] = ApplyPlaneRotation(H[k,i], H[k+1,i], cs[k], sn[k])
    
    cs[i],sn[i] = GeneratePlaneRotation(H[i,i], H[i+1,i], cs[i], sn[i]) 
    H[i,i],H[i+1,i] = ApplyPlaneRotation(H[i,i], H[i+1,i], cs[i], sn[i])
    s[i],s[i+1] = ApplyPlaneRotation(s[i], s[i+1], cs[i], sn[i])

    return H, cs, sn, s

def gmres_solver (Precond, E_hat, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, 
                    normal_xDev, normal_yDev, normal_zDev, xj, yj, zj, xi, yi, zi,  
                    xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                    xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, 
                    sizeTarDev, offsetMltDev, offsetIntDev, intPtrDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                    tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost,
                    offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, K, threshold, BSZ, GSZ, BlocksPerTwig, X, b, 
                    R, tol, max_iter, Cells, theta, Nm, II, JJ, KK, index, index_large, IndexDev, combII, combJJ, combKK, 
                    IImii, JJmjj, KKmkk, index_small, index_ptr, P, kappa, NCRIT, twig, eps):

    N = len(b)
    V = zeros((R+1, N), dtype=REAL)
    H = zeros((R+1,R), dtype=REAL)

    time_Vi = 0.
    time_Vk = 0.
    time_rotation = 0.
    time_lu = 0.
    time_update = 0.

    # Initializing varibles
    rel_resid = 1.
    cs, sn = zeros(N, dtype=REAL), zeros(N, dtype=REAL)

    iteration = 0

    b_norm = norm(b)

    time_eval  = 0.
    time_trans = 0.
    time_an    = 0.
    time_P2P   = 0.
    time_P2M   = 0.
    time_M2M   = 0.
    time_M2P   = 0.
    time_pack  = 0.
    

    '''
    A11 = zeros((N/2,N/2))
    A22 = zeros((N/2,N/2))
    for i in range(N/2):
        if i%50==0:
            print 'element %i'%i
        vec = zeros(N)
        vec[i]   = 1.
        vec[i+N/2] = 1.
        aux, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans, AI_int = gmres_dot(Precond, 
                    E_hat, vec, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, normal_xDev, normal_yDev, normal_zDev, xj, yj, zj, xi, yi, zi, 
                    xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                    xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, 
                    sizeTarDev, offsetMltDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                    tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost, 
                    offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, Cells, theta, Nm, II, JJ, KK, 
                    index, index_large, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, 
                    P, kappa, NCRIT, K, threshold, BSZ, GSZ, BlocksPerTwig, twig, eps, time_eval, time_P2P, 
                    time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans)
        A11[:,i] = aux[0:N/2]
        A22[:,i] = aux[N/2:N]

    savetxt('A11fast_500wp.txt',A11)
    savetxt('A22fast_500wp.txt',A22)
    quit()
    ''' 

    while (iteration < max_iter and rel_resid>=tol): # Outer iteration
        
        aux, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans, AI_int = gmres_dot(Precond, 
                    E_hat, X, vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, normal_xDev, normal_yDev, normal_zDev, xj, yj, zj, xi, yi, zi, 
                    xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                    xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, 
                    sizeTarDev, offsetMltDev, offsetIntDev, intPtrDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                    tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost, 
                    offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, Cells, theta, Nm, II, JJ, KK, 
                    index, index_large, IndexDev, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, index_ptr, 
                    P, kappa, NCRIT, K, threshold, BSZ, GSZ, BlocksPerTwig, twig, eps, time_eval, time_P2P, 
                    time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans)
        
        r = b - aux
        beta = norm(r)

        if iteration==0: 
            print 'Analytical integrals: %i of %i, %i'%(AI_int*2/len(X), len(X)/2, 100*4*AI_int/len(X)**2)+'%'

        V[0,:] = r[:]/beta
        if iteration==0:
            res_0 = b_norm

        s = zeros(R+1, dtype=REAL)
        s[0] = beta
        i = -1

        while (i+1<R and iteration+1<=max_iter): # Inner iteration
            i+=1 
            iteration+=1

            if iteration%10==0:
                print 'iteration: %i, residual %s'%(iteration,rel_resid)
                

            # Compute Vip1
            tic = time.time()
       
            Vip1, time_eval, time_P2P, time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans, AI_int = gmres_dot(Precond, 
                        E_hat, V[i,:], vertex, triangle, triHost, triDev, kHost, kDev, vertexHost, vertexDev, AreaHost, AreaDev, normal_xDev, normal_yDev, normal_zDev, xj, yj, zj, xi, yi, zi, 
                        xtHost, ytHost, ztHost, xsHost, ysHost, zsHost, xcHost, ycHost, zcHost, 
                        xtDev, ytDev, ztDev, xsDev, ysDev, zsDev, xcDev, ycDev, zcDev, 
                        sizeTarDev, offsetMltDev, offsetIntDev, intPtrDev, Pre0Dev, Pre1Dev, Pre2Dev, Pre3Dev, Pre0Host, Pre1Host, Pre2Host, Pre3Host,
                        tarPtr, srcPtr, offSrc, mltPtr, offMlt, offsetSrcHost, offsetSrcDev, offsetTarHost, sizeTarHost, 
                        offsetMltHost, Area, normal, xk, wk, xkDev, wkDev, Cells, theta, Nm, II, JJ, KK, 
                        index, index_large, IndexDev, combII, combJJ, combKK, IImii, JJmjj, KKmkk, index_small, index_ptr, 
                        P, kappa, NCRIT, K, threshold, BSZ, GSZ, BlocksPerTwig, twig, eps, time_eval, time_P2P, 
                        time_P2M, time_M2M, time_M2P, time_an, time_pack, time_trans)


            toc = time.time()
            time_Vi+=toc-tic

            tic = time.time()
            Vk = V[0:i+1,:]
            H[0:i+1,i] = dot(Vip1,transpose(Vk))

            # This ends up being slower than looping           
#            HVk = H[0:i+1,i]*transpose(Vk)
#            Vip1 -= HVk.sum(axis=1)

            for k in range(i+1):
                Vip1 -= H[k,i]*Vk[k] 
            toc = time.time()
            time_Vk+=toc-tic

            H[i+1,i] = norm(Vip1)
            V[i+1,:] = Vip1[:]/H[i+1,i]

            tic = time.time()
            H,cs,sn,s =  PlaneRotation(H, cs, sn, s, i, R)
            toc = time.time()
            time_rotation+=toc-tic

            rel_resid = abs(s[i+1])/b_norm

            if (i+1==R):
                print('Residual: %f. Restart...'%rel_resid)
            if rel_resid<=tol:
                break

        # Solve the triangular system
        tic = time.time()
        piv = arange(i+1)
        y = lu_solve((H[0:i+1,0:i+1], piv), s[0:i+1], trans=0)
        toc = time.time()
        time_lu+=toc-tic

        # Update solution
        tic = time.time()
        Vj = zeros(N)
        for j in range(i+1):
            # Compute Vj
            Vj[:] = V[j,:]
            X += y[j]*Vj
        toc = time.time()
        time_update+=toc-tic


#    print 'Time Vip1    : %fs'%time_Vi
#    print 'Time Vk      : %fs'%time_Vk
#    print 'Time rotation: %fs'%time_rotation
#    print 'Time lu      : %fs'%time_lu
#    print 'Time update  : %fs'%time_update
    print 'GMRES solve'
    print 'Converged after %i iterations to a residual of %s'%(iteration,rel_resid)
    print 'Time P2M          : %f'%time_P2M
    print 'Time M2M          : %f'%time_M2M
    print 'Time packing      : %f'%time_pack
    print 'Time data transfer: %f'%time_trans
    print 'Time evaluation   : %f'%time_eval
    print '\tTime M2P  : %f'%time_M2P
    print '\tTime P2P  : %f'%time_P2P
    print '\tTime analy: %f'%time_an
    print '------------------------------'
#    print 'Tolerance: %f, maximum iterations: %f'%(tol, max_iter)


    return X

"""
## Testing
xmin = -1.
xmax = 1.
N = 5000
h = (xmax-xmin)/(N-1)
x = arange(xmin, xmax+h/2, h)

A = zeros((N,N))
for i in range(N):
    A[i] = exp(-abs(x-x[i])**2/(2*h**2))

b = random.random(N)
x = zeros(N)
R = 50
max_iter = 5000
tol = 1e-8

tic = time.time()
x = gmres_solver(A, x, b, R, tol, max_iter)
toc = time.time()
print 'Time for my GMRES: %fs'%(toc-tic)

tic = time.time()
xs = solve(A, b)
toc = time.time()
print 'Time for stright solve: %fs'%(toc-tic)


tic = time.time()
xg = gmres(A, b, x, tol, R, max_iter)[0]
toc = time.time()
print 'Time for scipy GMRES: %fs'%(toc-tic)


error = sqrt(sum((xs-x)**2)/sum(xs**2))
print 'error: %s'%error
"""
