import numpy as np


def der1data(x, y, h, method):
    '''DER1DATA computes the first derivative of 
    sampled function (X,Y).
    DER1DATA(X,Y,H,METHOD) computes de first derivative 
    of the sampled function given as X and Y arrays. It 
    is assumed that, the abscisas X are equally spaced 
    with step H. The possible methods are:
    'f': forward differences with one step
    'b': backward differences with one step
    'c': central differences with one step'''
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    
    if method == 'f':
        df = (y[1:] - y[0:-1]) / h
        x = x[0:-1]
    elif method == 'b':
        df = (y[1:] - y[0:-1]) / h
        x = x[1:]
    elif method =='c':
        df = (y[2:] - y[0:-2]) / (2*h)
        x = x[1:-1]
    else:
        print('invalid method');
    
    return np.array([x, df])
    
def der2data(x, y, h, method):
    '''DER2DATA computes the second derivative of 
    sampled function (X,Y).
    DER2DATA(X,Y,H,METHOD) computes de second derivative 
    of the sampled function given as X and Y arrays. It 
    is assumed that, the abscisas X are equally spaced 
    with step H. The possible methods are:
    'f': forward differences with one step
    'b': backward differences with one step
    'c': central differences with one step'''
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    
    if method == 'f':
        df = (y[2:] - 2*y[1:-1]+y[0:-2]) / (h**2)
        x = x[0:-2]
    elif method == 'b':
        df = (y[2:] - 2*y[1:-1]+y[0:-2]) / (h**2)
        x = x[2:]
    elif method =='c':
        df = (y[2:] - 2*y[1:-1]+y[0:-2]) / (h**2)
        x = x[1:-1]
    else:
        print('invalid method');
    
    return np.array([x, df])
    
    
def der1(f, x, h, method):
    '''DER1 computes the first derivative of function F.
    DER1DATA(F,X,H,METHOD) computes de first derivative 
    of the function F at point X. The possible methods are:
    'f': forward differences with one step
    'b': backward differences with one step
    'c': central differences with one step.
    X could be an array or instead H could be an array,
    but they should be created using numpy.'''
	
    if method == 'f':
        df = (f(x + h) - f(x)) / h
    elif method == 'b':
        print(f(x) - f(x-h)) / h
    elif method =='c':
        df = (f(x + h) - f(x - h)) / (2.*h)
    else:
        print('invalid method');
        
    return df



# First derivative using O[h**2] central formula and a simple
# method to find the optimal step size
  
def der_min_1(f, x,tol= 10**(-10), h = 0.1):
    '''DER_MIN_1 computes the first derivative of function F.
    DER_MIN_1(F,X,tol,H,METHOD) computes de first derivative of the 
    function F at point X using the method of the minimum error. 
    An estimate of the initial step can be given'''
    
    c = 1.4

    dfold = (f(x + h) - f(x - h)) / (2.*h)
    h = h/c
    dfnew = (f(x + h) - f(x - h)) / (2.*h)
    
    err_old = 1e30
    err_new = abs(dfnew - dfold)

    while( err_old > err_new and err_old>tol ):
        
        dfold = dfnew
        h = h/c			
        dfnew = (f(x + h) - f(x - h)) / (2.*h)
        
        err_old = err_new
        err_new = abs(dfnew - dfold)

    return dfold, err_old

#Second derivative using O[h**2] central formula and a simple
# method to find the optimal step size

def der_min_2(f, x,tol= 10**(-10), h = 0.1):
    '''DER_MIN_2 computes the first derivative of function F.
    DER_MIN_2(F,X,tol,H,METHOD) computes de second derivative of the 
    function F at point X using the method of the minimum error. 
    An estimate of the initial step can be given'''
    
    c = 1.4

    dfold = (f(x + h) - 2*f(x) + f(x-h)) / (h**2)
    h = h/c
    dfnew = (f(x + h) - 2*f(x) + f(x-h)) / (h**2)
    
    err_old = 1e30
    err_new = abs(dfnew - dfold)

    while( err_old > err_new and err_old>tol ):
        
        dfold = dfnew
        h = h/c			
        dfnew = (f(x + h) - f(x - h)) / (2.*h)
        
        err_old = err_new
        err_new = abs(dfnew - dfold)

    return dfold, err_old

def central_difference(f, x, h):
    """Calculate the central difference derivative of f at x with step size h."""
    return (f(x + h) - f(x - h)) / (2 * h)

def drichardson(f, x, h0, c=2, max_iter=8, debug=False):
    """
    Compute the derivative of function f at x using Richardson extrapolation.
    """
    d = np.zeros([max_iter,max_iter], dtype=np.float64)
    d[0][0] = central_difference(f, x, h0)
    d_final = d[0][0]
    err = 1000
    
    for row in range(1, max_iter):
        h = h0 / (c ** row)
        d[row][0] = central_difference(f, x, h)
        
        for col in range(1, row + 1):
            d[row][col] = (c ** (2 * col) * d[row][col - 1] - d[row - 1][col - 1]) / (c ** (2 * col) - 1)
            err_max = max(abs(d[row][col] - d[row][col - 1]), abs(d[row][col] - d[row - 1][col - 1]))
            
            if err_max < err:
                err = err_max
                d_final = d[row][col]
            if col==row:
                if abs(d[row][col] - d[row-1][col - 1]) > 2 * err:
                    if debug:
                        print(d)
                    return [d_final, err]
    if debug:
        print(d)
    return [d_final, err]