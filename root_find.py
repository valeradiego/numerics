# Root finding basic solvers

# import modules
import warnings


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