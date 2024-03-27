# Polynomial interpolation

# import modules
import numpy as np 
import warnings 


# Lagrange Interpolation
def lagrange(x, y, xx):
    '''Computes the interpolated value at xx for the discrete 
    function given by (x, y) pairs using Lagrange interpolation.
    INPUT:
        x: abcisas of function to interpolate
        y: ordinates of function to interpolate 
        xx array or scalar to interpolate'''
    
    # Convert to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xx = np.asarray(xx, dtype=np.float64)
    
    # Convert scalar to array
    scalar = False
    if xx.ndim == 0:
        xx = xx[None]  # add new axis
        scalar = True
    
    n = len(x)
    nn = len(xx)
    sum = np.zeros_like(xx)
    
    for i in range(0, n):
        l = np.ones(nn)
        for j in range(0, n):
            if (i != j):
                l = l * (xx - x[j]) / (x[i] - x[j])
        sum = sum + y[i] * l
    
    # Scalar case
    if scalar:
        return sum[0]
    
    return sum



# Piecewise intepolation

# Spline function: computes second derivatives
def spline(x, y, yp1=None, ypn=None):
    """
    spline computes the second derivatives at all points x of the discrete
    function y(x) to be used for spline interpolation.
    INPUT:
        x: abscisas of discrete function
        y: ordinates of discrete function
        yp1: first derivative at first point, None for Natural SPLINE
        ypn: first derivative at last point, None for Natural SPLINE
    OUTPUT
        y2: second derivatives at all points
    """      
    
    # Convert to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
        
    # Initialize internal numpy arrays
    u = np.empty_like(x)
    y2 = np.empty_like(x)
        
    # Condition for the initial point    
    if (yp1 == None):
        # natural spline
        y2[0] = 0.
        u[0] = 0.
    else:
        # first derivative
        y2[0] = -0.5
        u[0]  = (3./(x[1] - x[0]))*((y[1] - y[0])/(x[1] - x[0]) - yp1)
    
    # Condition for the last point 
    if (ypn == None):
        # natural spline
        qn = 0.
        un = 0.
    else:
        # first derivative
        qn = 0.5
        un = (3./(x[n-1] - x[n-2]))*(ypn - (y[n-1] - y[n-2])/(x[n-1] - x[n-2]))
    
    # Setup tridiagonal equations
    for i in range(1, n-1):
        sig = (x[i] - x[i-1])/(x[i+1] - x[i-1])
        
        p = sig*y2[i-1] + 2.
        
        y2[i] = (sig - 1.)/p
        
        u[i] = (6.*((y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i]-y[i-1])/
         (x[i] - x[i-1]))/(x[i+1] - x[i-1]) - sig*u[i-1])/p
    
    
    # Solve tridiagonal system for second derivatives
    y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.)
    
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k]*y2[k+1] + u[k]
        
    return y2
   
   
# Splint function: interpolate at a given point
def splint(xa, ya, y2a, x):
    """
    splint makes the spline interpolation of discrete function y(x) using 
    the second derivatives computed with spline algorithm.
    INPUT:
        x: abscisas of discrete function
        y: ordinates of discrete function
        y2: second derivatives at all points 
    OUTPUT:
        x: interpolated value
    """    
    
    # Convert to numpy arrays
    xa = np.asarray(xa, dtype=np.float64)
    ya = np.asarray(ya, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
        
    # Convert scalar to array
    scalar = False
    if x.ndim == 0:
        x = x[None]  # add new axis
        scalar = True
    
    # Allocate y
    y = np.empty_like(x)
    
    n = len(xa)
    for i in range(len(x)):
        # Find interval that contains x
        klo = 0
        khi = n - 1
        while (khi - klo > 1):
            k = int((khi + klo)/2)
            if(xa[k] > x[i]):
                khi = k
            else:
                klo = k
        
        # Evaluate cubic polynomial in x
        h = xa[khi] - xa[klo]
        
        if (h == 0.): 
            warnings.warn('splint::err: all xa must be different')
            return 0
        
        a = (xa[khi] - x[i])/h  # lagrange A(x)
        b = (x[i] - xa[klo])/h  # Lagrange B(x)
        
        y[i] = a*ya[klo] + b*ya[khi] + ((a**3 - a)*y2a[klo] + (b**3 - b)*y2a[khi])*(h**2)/6.
    
    # Scalar case
    if scalar:
        return y[0]
    
    return y
      
# Splint_der function: interpolate derivate at a given point
def splint_der(xa, ya, y2a, x):
    """
    splint_der makes the spline interpolation of the derivatie  of discrete function y'(x) using 
    the second derivatives computed with spline algorithm.
    INPUT:
        x: abscisas of discrete function
        y: ordinates of discrete function
        y2: second derivatives at all points 
    OUTPUT:
        y': interpolated value
    """    
    
    # Convert to numpy arrays
    xa = np.asarray(xa, dtype=np.float64)
    ya = np.asarray(ya, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
        
    # Convert scalar to array
    scalar = False
    if x.ndim == 0:
        x = x[None]  # add new axis
        scalar = True
    
    # Allocate y_der
    y_der = np.empty_like(x)
    
    n = len(xa)
    for i in range(len(x)):
        # Find interval that contains x
        klo = 0
        khi = n - 1
        while (khi - klo > 1):
            k = int((khi + klo)/2)
            if(xa[k] > x[i]):
                khi = k
            else:
                klo = k
        
        # Evaluate cubic polynomial in x
        h = xa[khi] - xa[klo]
        
        if (h == 0.): 
            warnings.warn('splint::err: all xa must be different')
            return 0
        
        a = (xa[khi] - x[i])/h  # lagrange A(x)
        b = (x[i] - xa[klo])/h  # Lagrange B(x)
        
        y_der[i]=(-1/h)*ya[klo] +(1/h)*ya[khi] +((3*a**2*(-1/h)-(-1/h))*y2a[klo] +(3*b**2*(1/h)-(1/h))*y2a[khi])*(h**2)/6 #derivada de y con respecto a x
    # Scalar case
    if scalar:
        return y_der[0]
    
    return y_der  
def gauss_solve(A, b):
    n = len(A)
    Ab = np.concatenate((A, b), axis=1)  # Augmented matrix

    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]

        pivot = Ab[i, i]
        if pivot == 0:
            raise ValueError("No unique solution exists.")

        Ab[i] /= pivot
        for j in range(i + 1, n):
            Ab[j] -= Ab[j, i] * Ab[i]

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = Ab[n - 1, n]
    for i in range(n - 2, -1, -1):
        x[i] = Ab[i, n] - np.dot(Ab[i, i + 1:n], x[i + 1:n])

    return x
def Horner(a,x):
	"""
	Esta funcion halla el valor en un polinomio aplicando el algoritmo de horner donde a=(a_0,a_1,..,a_n)"""
	a=asarray(a, dtype=np.float64)
	x=asarray(x, dtype=np.float64)
	if a.size==1:
		return [a[0]]
	i=a.size-1
	sum=a[i]*x+a[i-1]
	while i>1:
		i=i-1
		sum=sum*x+a[i-1]
	return sum
def inverse_det(A):
    
    n = len(A)
    I = np.identity(n)
    inverse = gauss_solve2(A, I)
    det = 1
    
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(A[i:, i])) + i
        A[[i, max_row]] = A[[max_row, i]]      
        if i != max_row: det *=(-1) 
        

        pivot = A[i, i]
        if pivot == 0:
            raise ValueError("No unique solution exists.")

        A[i]/=pivot
        det*=pivot
        for j in range(i + 1, n):
            A[j]-=A[j, i]*A[i]
            
    for i in range(n):
        det *= A[i, i]
    
    return inverse, det
def vander_interp(xa,ya,x):
    #definimos una matriz con las potencias de x
    xa=np.asarray(xa, dtype=np.float64)
    ya=np.transpose(ya)
    Avander=np.zeros((len(xa),len(xa)))
    for i in range(len(xa)):
        for j in range(len(xa)):
            Avander[i][j]=xa[i]**j
    #usamos gauss para hallar el arreglo de coeficientes
    coef=gauss_solve(Avander, ya)
    #usamos horner para evaluar x en estos coeficientes
    evaluacion= Horner(coef,x)
    #hallamos el condition numbre
    inv=inverse_det(Avander)[0]
    k=np.linalg.norm(Avander)*np.linalg.norm(inv)
    print(k)
    return evaluacion