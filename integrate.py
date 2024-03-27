import numpy as np

# integration using extended trapezoidal rule
def trapz_n(f, a, b, s, n):
    ''' 
    trapz_n: integrates f from a to b using the trapezoidal
    rule with N = 2**(n - 1) intervals.
    * To compute the integral with some n, it is required that 
      the integral has been computed for previous (n - 1) and the 
      corresponding value of s been saved to be used as input.      
    * This means that s is a input and output variable. 
    * Then trapz_n must be called with succesive values n = 1, 2, ... 
      saving the each corresponding s for next calculation.    
    '''
    if (n == 1):
        # just trapezoidal with one interval
        s = 0.5*(b - a)*(f(a) + f(b))
    else:
        # adjust new step
        new = 2**(n - 2)   # number of new points
        h = (b - a)/new    # step in previous iteration
        
        # compute the contribution of new points 
        sum = 0.
        for j in range(1, new + 1):
            x = a + (j - 0.5)*h
            sum = sum + f(x)

        # update new integral
        s = 0.5*(s + h*sum)

    return s


# integration using an iterative trapezoidal algorithm
def trapz(f, a, b, eps=10**(-10)):

    nmax = 24
    zero = 10.**(-15)
    
    h = (b - a)
    sold = 0.
    snew = 0.5*h*(f(a) + f(b))
    
    for n in range(2,nmax+1):
        
        term = 0.
        sold = snew
        
        # compute the contribution of new points
        for k in range(1, 2**(n - 2) + 1):
            term = term + f(a + (k - 0.5)*h)
        
        snew = 0.5*(sold + h*term)
        h = 0.5*h  # step for next iteration
        
        # print("n: %2d, integral: %.16f, relative error: %e"%(n, snew, abs(snew-sold)/abs(snew)))
        if (n > 6):
            if( abs(snew-sold) < abs(snew)*eps or (abs(snew) <= zero and abs(sold) <= zero) ):
                return (n, snew)

    # To return a value anyway we give the value in the
    # last iteration.
    print('trapz: nmax reached without convergence, result could be inaccurate.')
    return (n, snew)

def trap_ext(f,a,b,N):
    '''te calcula la integral con trapezoidal extendida de una funcion desde un punto a hasta un punto b en N intervalos'''   
    h=(a-b)/N
    s=(f(a)+f(b))/2
    for i in range (1,N):
        s+=f(a+i*h)
    return(h*s)

def simp_ext(f,a,b,N):
    '''te calcula la integral simpson extendida de una funcion desde un punto a hasta un punto b en N intervalos'''   
    h=(a-b)/N
    s=f(a)+f(b)
    for i in range (1,N):
        if i%2!=0:
            s+=4*f(a+i*h)
        else:
            s+=2*f(a+i*h)
    return(h/3*s)

def simp(f, a, b, eps=10**(-10)):
    nmax = 24
    zero = 10.**(-15)
    
    h = (b - a)
    newtrap = 0.5*h*(f(a) + f(b))
    simp=0
    for n in range(2,nmax+1):
        
        term = 0.
        oldtrap = newtrap
        oldsimp=simp
        # compute the contribution of new points
        for k in range(1, 2**(n - 2) + 1):
            term = term + f(a + (k - 0.5)*h)
        
        newtrap = 0.5*(oldtrap + h*term)
        simp=4/3*newtrap-1/3*oldtrap
        h = 0.5*h  # step for next iteration
        
        # print("n: %2d, integral: %.16f, relative error: %e"%(n, snew, abs(snew-sold)/abs(snew)))
        if (n > 6):
            if( abs(simp-oldsimp) < abs(simp)*eps or (abs(simp) <= zero and abs(oldsimp) <= zero) ):
                return (n, simp)

    # To return a value anyway we give the value in the
    # last iteration.
    print('simp: nmax reached without convergence, result could be inaccurate.')
    return (n, simp)


