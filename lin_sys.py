import numpy as np

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

def gauss_jordan(A, b):
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

    for j in range(n-2,-1,-1):
        for i in range(n-1,j,-1):
            Ab[j]-=Ab[j,i]*Ab[i]
    # Back substitution
    x = np.zeros(n)
    x[n - 1] = Ab[n - 1, n]
    for i in range(n - 2, -1, -1):
        x[i] = Ab[i, n]
    return x

def gauss_solve2(A, B):
    n = len(A)
    X=[]
    for i in range(n):
        bi = np.reshape(B[:,i],[n,i])   
        Ab = np.concatenate((A, bi), axis=1)  # Augmented matrix

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
    
    X.append(x)
    X=np.array(X)
    X=X.transpose()
    return X

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
def tridiag_solve(a,b,c,r):
    '''
    a: diagonal de abajo
    b: diagonal del medio
    c: diagonal de arriba
    r: vector inhomogeneo
    '''
    
    n = len(r)
    beta = np.zeros(n ,float)
    rho = np.zeros(n, float)
    x = np.zeros(n,float)
    beta[0] = b[0]
    rho[0] = r[0]
    for j in range(1, n):
        beta[j] = b[j] - (a[j-1]/beta[j-1]) * c[j-1]
    for j in range(1,n):
        rho[j] = r[j] - (a[j-1]/beta[j-1]) * rho[j-1]
    x[n-1] = rho[n-1]/beta[n-1]
    for j in range(2, n+1):
        x[n-j] = (rho[n-j]-c[n-j]*x[n-j+1])/beta[n-j]
        
    return x