import numpy as np

def euler_1d(f, y0, x0, xn, n):
    """
    euler_1d integrates the edo dx/dt = f(t,x)
    input:
        f : integration function
        y0: initial condition
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros(n);
    
    x = np.linspace(x0, xn, n)
    y[0] = y0
    
    for i in range(1, n):
        y[i] = y[i-1] + h * f(x[i-1], y[i-1])
        
    return (np.array(x), np.array(y))
    
def eulermod_1d(f, y0, x0, xn, n):
    """
    eulermod_1d integrates the edo dx/dt = f(t,x)
    input:
        f : integration function
        y0: initial condition
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros(n);
    
    x = np.linspace(x0, xn, n)
    y[0] = y0
    
    for i in range(1, n):
        y[i] = y[i-1] + h * f(x[i-1]+0.5*h, y[i-1]+0.5*h*f(x[i-1],y[i-1]))
        
    return (np.array(x), np.array(y)) 

def ralston_1d(f, y0, x0, xn, n):
    """
    ralston_1d integrates the edo dx/dt = f(t,x)
    input:
        f : integration function
        y0: initial condition
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros(n);
    
    x = np.linspace(x0, xn, n)
    y[0] = y0
    
    for i in range(1, n):
        k1 = f(x[i-1], y[i-1])
        k2 = f(x[i-1]+(2/3)*h,y[i-1]+(2/3)*h*k1)
        y[i] = y[i-1]+(h/4)*(k1+3*k2)
        
    return (np.array(x), np.array(y)) 

def eulermej_1d(f, y0, x0, xn, n):
    """
    eulermod_1d integrates the edo dx/dt = f(t,x)
    input:
        f : integration function
        y0: initial condition
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros(n);
    
    x = np.linspace(x0, xn, n)
    y[0] = y0
    
    for i in range(1, n):
        y[i] = y[i-1] + 0.5*h*(f(x[i-1], y[i-1]) +f(x[i], y[i-1] +h*f(x[i-1], y[i-1])))
        
    return (np.array(x), np.array(y)) 

def euler_nd(f, y0, x0, xn, n):
    """
    euler_nd integrates the edo dy/dt = f(t, y)
    input:
        f : integration function (array valued)
        y0: initial condition (array)
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros([np.size(y0), n]);

    x = np.linspace(x0, xn, n)
    y[:, 0] = y0

    for i in range(1, n):
        y[:, i] = y[:, i-1] + h * f(x[i-1], y[:, i-1])
        
    return (np.array(x), np.array(y))

def eulermod_nd(f, y0, x0, xn, n):
    """
    eulermod_nd integrates the edo dy/dt = f(t, y)
    input:
        f : integration function (array valued)
        y0: initial condition (array)
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros([np.size(y0), n]);

    x = np.linspace(x0, xn, n)
    y[:, 0] = y0

    for i in range(1, n):
        y[:, i] = y[:, i-1] + h * f(x[i-1] + 0.5*h, y[:, i-1] + 0.5*h*f(x[i-1], y[:, i-1]))
        
    return (np.array(x), np.array(y))

def eulermej_nd(f, y0, x0, xn, n):
    """
    eulermod_nd integrates the edo dy/dt = f(t, y)
    input:
        f : integration function (array valued)
        y0: initial condition (array)
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros([np.size(y0), n]);

    x = np.linspace(x0, xn, n)
    y[:, 0] = y0

    for i in range(1, n):
        y[:, i] = y[:, i-1] +0.5*h*(f(x[i-1], y[:, i-1]) + f(x[i], y[:, i-1] + h*f(x[i-1], y[:, i-1])))
        
    return (np.array(x), np.array(y)) 

def ralston_nd(f, y0, x0, xn, n):
    """
    ralston_nd integrates the edo dy/dt = f(t, y)
    input:
        f : integration function (array valued)
        y0: initial condition (array)
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    x = np.zeros(n);
    y = np.zeros([np.size(y0), n]);

    x = np.linspace(x0, xn, n)
    y[:, 0] = y0

    for i in range(1, n):
        k1 = f(x[i-1], y[:, i-1])
        k2 = f(x[i-1] + (2/3)*h, y[:, i-1] + (2/3)*h*k1)
        y[:, i] = y[:, i-1] + (h/4) * (k1 + 3*k2)
    return (np.array(x), np.array(y)) 

def rk4_nd(f, y0, x0, xn, n):
    """
    rk4_nd integrates the edo dy/dt = f(t, y)
    input:
        f : integration function (array valued)
        y0: initial condition (array)
        x0: initial point
        xn: final point
        n : number of points
    output:
        x : numpy array of abscisas
        y : numpy array of ordinates
    """
    h = (xn - x0)/(n - 1)
    y = np.zeros([np.size(y0), n])
    x = np.linspace(x0, xn, n)
    y[:, 0] = y0

    for i in range(1, n):
        k1 = f(x[i-1], y[:, i-1])
        k2 = f(x[i-1] + 0.5*h, y[:, i-1] + h*(k1/2))
        k3 = f(x[i-1] + 0.5*h, y[:, i-1] + h*(k2/2))
        k4 = f(x[i-1] + h, y[:, i-1] + h*k3)
        y[:, i] = y[:, i-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return (np.array(x), np.array(y)) 
# RK4 Simple Adaptative Algorithm

def rk4_1d_adaptive(f, y0, x0, xf, rel_tol=1e-8, abs_tol=1e-8, hmin=1e-4, hmax=1e-1):
    
    # Factor to reduce tolerance to min
    SAFE = 1/10
    
    # initial values
    x = x0
    y = y0
    h = hmax
    
    # initial ouput lists
    x_arr = [x]
    y_arr = [y]
    h_arr = [hmax]
    
    # can not use x <= xf !
    while (x < xf):
      
        # Adjust last step if it exceeds xf
        if x + h > xf:
            h = xf - x  

        # Set min and max values for h        
        if h < hmin:
            h = hmin
        elif h > hmax:
            h = hmax
            
        # Do a single step
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        
        y_sgl = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    

        # Estimate absolute error
        error = abs(y_hlf - y_sgl)
        tol = (abs(y_hlf) + h*abs(f(x, y)))*rel_tol + abs_tol
        
        if (error > tol) and h > hmin:
            # Error is too large, decrease step
            h /= 2 
            
        else:
            # Accept the step
            x +=h
            y = y_hlf             
            x_arr.append(x)
            y_arr.append(y)
            h_arr.append(h)
            
            # Small error, increase step size
            if (error < SAFE*tol):
                h *= 2 
            
    return np.array(x_arr), np.array(y_arr), np.array(h_arr)    
    