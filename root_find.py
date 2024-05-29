# Root finding basic solvers

# import modules
import warnings
import numpy as np

# Bisection's Method
def bisection(func, x1, x2, tol=1.0e-15,num_iter=False):
    """
    Uses the bisection method to find a value x between x1 and x2
    for which f(x) = 0, to within the tolerance given.
    Defalt tolerance: 1.0e-15.
    """
    
    JMAX = 100
    fmid = func(x2)
    f = func(x1)
    
    if f*fmid >= 0.: 
        warnings.warn('bisection: root must be bracketed')
        return 
        
    if f < 0.:
        root = x1
        dx = x2-x1
    else:
        root = x2
        dx = x1 - x2
        
    for j in range(1,JMAX+1):
        dx = dx*.5
        xmid = root + dx
        fmid = func(xmid)
        
        if fmid <= 0.:
            root = xmid
        
        # Convergence test or zero in midpoint: 
        if abs(dx) < abs(root)*tol or fmid == 0.0:
            if num_iter==True:
                return (root,j)
            return root
        
    warnings.warn('bisection: too many bisections')
    if num_iter==True:
        return (root,JMAX)
    return root
    

# Newton's Method  
def newton(f, df, x, tol=1.0e-15,num_iter=False):
    """
    Uses Newton's method to find a value x near "x" for which f(x) = 0, 
    to within the tolerance given.
    
    INPUT: 
        Default tolerance: 1.0e-15
        Functions: f(x) and f'(x).
    """
    
    MAXIT = 20;
    root = x;
    
    for j in range(1, MAXIT + 1):
        
        dx = f(root) / df(root);
        root = root - dx;
        
        # Convergence is tested using relative error between
        # consecutive approximations.
        if (abs(dx) <= abs(root)*tol or f(root) == 0):
            if num_iter==True:
                return (root,j)
            return root
        
    warnings.warn('newton::err: max number of iterations reached.')  
    if num_iter==True:
        return (root,MAXIT)
    return root
    
#Metodo de la secante 
def secant(f,x1,x2,tol=1.0e-15,num_iter=False):    
    """
    Uses the secant method to find a value x between x1 and x2
    for which f(x) = 0, to within the tolerance given.
    Defalt tolerance: 1.0e-15.
    """
    MAXIT = 100
    fl=f(x1)
    frt=f(x2)

    if fl*frt >= 0.: 
        warnings.warn('secant: root must be bracketed')
        return 
    # rt cerca del extremo donde f es menor 
    if abs(fl)<abs(frt):
        root=x1
        xl=x2
        (fl,frt)=(frt,fl)
    else:
        root=x2
        xl=x1
    for j in range(1,MAXIT+1):
        dx=(xl-root)/(frt-fl)*frt
        xl=root
        fl=frt
        root=root+dx
        frt=f(root)
        if (abs(dx)<= abs(root)*tol or frt==0):
            if num_iter==True:
                return (root,j)
            return root
    warnings.warn('newton::err: max number of iterations reached.')  
    if num_iter==True:
        return (root,MAXIT)
    return root
#Metodo de falsa posicion
def false_position(f,x1,x2,tol=1.0e-15,num_iter=False):    
    """
    Uses the method of false position to find a value x between x1 and x2
    for which f(x) = 0, to within the tolerance given.
    Defalt tolerance: 1.0e-15.
    """
    MAXIT = 100
    fl=f(x1)
    fh=f(x2)
    if fl*fh >= 0.: 
        warnings.warn('false_position: root must be bracketed')
        return  
    if fl<0:
        xl=x1
        xh=x2
    else:
        xl=x2
        xh=x1
        (fl,fh)=(fh,fl)
    dx=xh-xl
    for j in range(1,MAXIT+1):
        root=xl+dx*fl/(fl-fh)
        frt=f(root)
        if(frt<0):
            dif=xl-root
            xl=root
            fl=frt
        else:
            dif=xh-root
            xh=root
            fh=frt
        dx=xh-xl
        if (abs(dif)<= abs(root)*tol or frt==0):
            if num_iter==True:
                return (root,j)
            return root
    warnings.warn('newton::err: max number of iterations reached.') 
    if num_iter==True:
        return (root,MAXIT) 
    return root

def newton_bisection(func, dfunc, x1, x2, tol=1.0e-15):
    MAXIT = 1000  # Número máximo de iteraciones
    f1 = func(x1)
    f2 = func(x2)

    if f1 < 0.0:
        xl, xh = x1, x2  # dirección de búsqueda, de xl a xh
    else:
        xl, xh = x2, x1
        f1, f2 = f2, f1

    rt = (x1 + x2) / 2.0
    f = func(rt)
    df = dfunc(rt)
    dx = xh - xl  # tamaño de paso, anterior y actual
    dxold = dx

    for i in range(MAXIT):
        if (rt < xl and rt > xh and xh>xl) or (rt<xh and rt>xl and xl>xh) or (abs(f/df)>abs(dx/2)):
            dxold = dx
            dx = 0.5 * (xh - xl)
            rt += dx  # bisección
        else:
            dxold = dx
            dx = f / df
            rt -= dx  # newton

        if abs(dx) < tol * abs(rt) or f == 0:
            return rt  # convergencia

        f = func(rt)
        df = dfunc(rt)

        if f < 0.0:
            xl = rt
        else:
            xh = rt

    print("Máximo número de iteraciones alcanzado")
    return rt

def solve_system(fvec, jacb, xvec, tol=1.0e-15, KMAX=1000):
    for k in range(1, KMAX + 1):
        f = fvec(xvec)
        J = jacb(xvec)
        
        # Resolver el sistema lineal J * dx = -f
        dx = np.linalg.solve(J, -f)
        
        # Actualizar la raíz
        xvec = xvec + dx
        
        # Comprobar la convergencia
        if np.sum(np.abs(dx)) <= tol:
            print(f"Convergencia alcanzada en {k} iteraciones.")
            return xvec
        
    print("Máximo número de iteraciones alcanzado")
    return xvec

import sympy as sp

def symbolic_jacobian(fvec, vars):
    """
    Calcula el Jacobiano de la función vectorial fvec simbólicamente.
    
    Parámetros:
    fvec : list of sympy expressions
        Lista de expresiones simbólicas que definen la función vectorial.
    vars : list of sympy symbols
        Lista de variables simbólicas.
        
    Retorna:
    jac : sympy.Matrix
        Matriz Jacobiana evaluada simbólicamente.
    """
    jacobian_matrix = sp.Matrix(fvec).jacobian(vars)
    return jacobian_matrix

# Brent's Method    
def brent(f, a, b, rel_tol=4.0e-16, abs_tol=2.e-12):
    """
    Brent's algorithm for root finding.
    
    Arguments:
    - f: The function for which to find the root.
    - a, b: Initial bracketing interval [a, b].
    - rel_tol: relative toleranece.
    - abs_tol: relative toleranece.
    - max_iterations: Maximum number of iterations.
    
    Returns:
    - The approximate root of the function.
    """

    fa = f(a)
    fb = f(b)
    c = a
    fc = fa
    e = b - a # interval size
    d = e     # step dx
    
    while (True):       
        # root between b and c
        # if root near c: move c,a to b, b to c 
        # else (root near b): don't do anything
        if (abs(fc) < abs(fb)):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        # compute tolerance
        tol = 2.0*rel_tol*abs(b) + abs_tol
        # half interval size
        m = 0.5*(c - b)

        # convergence
        if(abs(m) <= tol or fb == 0.0):
            break

        if (abs(e) < tol or abs(fa) <= abs(fb)):
            # take bisection
            e = m
            d = e # new step dx
        else:
            s = fb/fa
            if (a == c):
                # attemp linear interpolation
                p = 2.0*m*s
                q = 1.0 - s
            else:
                # attemp inverse quadratic interpolation
                q = fa/fc
                r = fb/fc
                p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0))
                q = (q - 1.0)*(r - 1.0)*(s - 1.0)
                
            if(0.0 < p):
                q = -q
            else:
                p = -p

            s = e
                
            # check region of interpolation
            if (2.0*p < 3.0*m*q - abs(tol*q) and p < abs(0.5*s*q)):
                # accept interpolation
                e = d  # store previous d
                d = p/q # new step dx
            else:
                # use bisection
                e = m
                d = e # new step dx

        # move a to b (old root)
        a = b
        fa = fb

        # update b (new root)
        if (tol < abs(d)):
            # with step d
            b = b + d
        elif(0.0 < m):
            # min value of d is tol
            b = b + tol
        else:
            # min value of d is tol
            b = b - tol
        fb = f(b)

        # if root between b and a: move c to a
        # else (root between c and b): don't do anything
        if ((0.0 < fb and 0.0 < fc) or (fb <= 0.0 and fc <= 0.0)):
            c = a
            fc = fa
            e = b - c   # new interval size
            d = e       # new step dx

    return b 

def central_difference(f, x, h):
    """Calculate the central difference derivative of f at x with step size h."""
    return (f(x + h) - f(x - h)) / (2 * h)

def richardson_extrapolation(f, x, h0, c=2, tolerance=1e-6, max_iter=8):
    """
    Compute the derivative of function f at x using Richardson extrapolation.
    """
    d = np.zeros([8,8])
    d[0][0] = central_difference(f, x, h0)
    err = tolerance
    
    for row in range(1, max_iter + 1):
        h = h0 / (c ** row)
        d[row][0] = central_difference(f, x, h)
        
        for col in range(1, row + 1):
            d[row][col] = (c ** (2 * col) * d[row][col - 1] - d[row - 1][col - 1]) / (c ** (2 * col) - 1)
            err_max = max(abs(d[row][col] - d[row][col - 1]), abs(d[row][col] - d[row - 1][col - 1]))
            
            if err_max < err:
                err = err_max
                d_final = d[row][col]
            
            if abs(d[row][col] - d[row][col - 1]) > 2 * err:
                return d_final
    print(d)
    return d_final
