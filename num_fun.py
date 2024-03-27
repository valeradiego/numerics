# import modules

from math import log
from numpy import sqrt
from numpy import vectorize
from numpy import asarray
from numpy import size
import numpy as np
import warnings

# Quadratic Equation Solver
# a*x**2+b*x+c 0

_w_e2solve_infsol = 'eq2solve@infsol: a=b=c=0, equation has infinite solutions'
_w_e2solve_nosol = 'eq2solve@nosol: a=b=0 and c!=0, equation has no solution'
_w_e2solve_lineq = 'eq2solve@lineq: a=0, equation is a linear equation'
_w_e2solve_cmplx = 'eq2solve@cmplx: equation has complex solutions'

def eq2solve(a, b, c):

	if a == 0.:   # calculation or input?
		if b == 0.:
			if c == 0. :
				warnings.warn(_w_e2solve_infsol)
				return []
			warnings.warn(_w_e2solve_nosol)
			return []
		else:
			warnings.warn(_w_e2solve_lineq)
			return [-c/b]
	else:
		delta = b*b - 4.*a*c;

		if delta < 0.:
			warnings.warn(_w_e2solve_cmplx);
			x1 = (-b - 1.j * sqrt(abs(delta)))/(2.*a);
			x2 = (-b + 1.j * sqrt(abs(delta)))/(2.*a);
			return [x1, x2]
		
		delta = sqrt(delta);
		
		if abs(delta) <= 10**(-10):  # better than delta == 0. ?
			# delta = 0, un par de soluciones iguales;
			x1 = -b/(2.*a)
			x2 = x1
			return [x1, x2]
		elif b < 0.:		
			x1 = (-b + delta) / (2.*a)
			x2 = (2.*c) / (-b + delta)
			return [x1, x2]
		elif b > 0.:
			x1 = (-b-delta) / (2.*a)
			x2 = (2.*c) / (-b - delta)
			return [x1, x2] 
		else:
			# b = 0, un par de soluciones con signos opuestos
			x1 = sqrt(-c/a)
			x2 = -x1
			return [x1, x2]
            
            
                        

        
# sqrtHeron: Computes the sqrt root of A using ancient algorithm 

# Case 1: Iterating n times, check convergence visually

def sqrtHeronTrivial(A):
	x = 1.;
	print('i = %d, root = %18.16f' % (0, x))
    
	for i in range(1,7):
		x = 0.5*(x + A/x);
        # check convergence visually
		print('i = %d, root = %18.16f' % (i, x))

	return x

# Case 2: Checking convergence automatically
    
def sqrtHeron(A, prec=16):

	delta = 10.**(-prec)
	old = 0.
	new = 1.
	count = 0

	while(abs(new-old) > delta * abs(new)):
		old = new
		new = 0.5*(new + A/new)
		count = count + 1
		
	return new, count



# Compute exp(x) using Tylor's series

def exp(x, eps=1.e-16):
    """
    This function computes the exponential function using the Taylor 
    series. It works for a scalar input x only.
	"""
	
    KMAX = 10000
    k = 1
    term = 1.
    oldsum = -1.
    newsum = 0.
	
    while ((abs(newsum-oldsum) > 0.5*abs(newsum)*eps) and (k <= KMAX)):
        oldsum = newsum
        newsum = newsum + term
        # print('tn: ', term)   # uncomment only for debug
        term = term * (x / k)
        k = k + 1
        
    if (KMAX == k-1):
        warnings.warn('Max number of iterations reached. Result could be inaccurate.')
        
    return newsum   
    
#Hallamos el coeficiente binomial C(n,k) por definicion:
def bin1(n,k):
	"""
	Esta funcion halla el coeficiente binomial aplicando la formula n!/(k!*(n-k)!)
	""" 

	if (n==0 and k==0) or n<0 or k<0:
		warnings.warn('indeterminacion')
		return []
	if n==0:
		return 0
	if k==0:
		return 1
	else:
		mult= n/k
		while k > 1:
			n=n-1
			k=k-1
			mult=mult*n/k
	return mult

#Hallamos el coeficiente binomial usando logaritmos:
def bin2(n,k):
	"""
	Esta funcion halla el coeficiente binomial aplicando exponenciales y logaritmos
	""" 

	if (n==0 and k==0) or n<0 or k<0:
		warnings.warn('indeterminacion')
		return []
	if n==0:
		return 0
	if k==0:
		return 1
	else:
		if k > n/2 : #ahorramos operaciones si usamos esta propiedad C(n,k)=C(n,(n-k))
			k=n-k
		sum= log(n)-log(k)
		while k > 1:
			n=n-1
			k=k-1
			sum= sum + log(n)-log(k)
		
	return exp(sum)
# Hallamos un valor de un polinomio aplicando el algoritmo de horner
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
# Hallamos el valor que describe la division entre dos polinomios evaluados en un array x usando Horner
def rat_eval(P1,P2,x):
	valor_P1=Horner(P1,x)
	valor_P2=Horner(P2,x)
	return valor_P1/valor_P2

#Hallamos la función de Bessel esférica para un valor de x (array) y l a partir del algoritmo de recurrencia de Bessel
def Bessel_algorithm(x,l):
	"""
	Esta función halla la función de Bessel esférica para un valor de x (array) y l a partir del algoritmo de recurrencia de Bessel (Forward recursive)"""
	x=asarray(x,dtype=np.float64)
	
	j0=np.sin(x)/x
	j1=np.sin(x)/x**2-np.cos(x)/x
	jnew=0
	for i in range(1,l):
		jnew=(2*i+1)/x*j1-j0
		j0=j1
		j1=jnew
	return jnew
#Hallamos la función de Bessel esférica para un valor de x (array) y l a partir del algoritmo de recurrencia de Miller
def Miller_algorithm(x, l):
	"""
	Esta función halla la función de Bessel esférica para un valor de x (array) y l a partir del algoritmo de recurrencia de Miller (Downward recursive)
    """
	x=asarray(x,dtype=np.float64)

	l_ini=round(l+3*sqrt(l))
	j_ini=0
	j_sec=1
	jnew=0

	for i in range(l_ini-1,l,-1):
		jnew=(2*i+1)/x*j_sec-j_ini
		j_ini=j_sec
		j_sec=jnew

	value=jnew

	for i in range(l,0,-1):
		jnew=(2*i+1)/x*j_sec-j_ini
		j_ini=j_sec
		j_sec=jnew
	factor=jnew/(np.sin(x)/x)
	return value/factor