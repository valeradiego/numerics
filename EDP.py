import numpy as np

def sor_itr(v, den, w, eps, save_iter=False, show_iter=False):
    
    '''
    SOR iterations for two-dimensional poisson PDE
         
         laplacian(V(x, y)) = -rho(x, y)
         
    w: relaxation parameter
    eps: absolute tolerance
    '''
    
    max_iter = 1000
    
    nd1, nd2 = v.shape


    if (not save_iter):
    
        for iter in range(max_iter):

            err = 0.0
        
            # we advance by rows in the stencil and it gives the proper formula for SOR
            for i in range(1, nd1 - 1):
                for j in range(1, nd2 - 1):
                    # Calculate the difference of V(x_i, y_j) between two iterations
                    diff = (den[i, j] + v[i-1, j] + v[i+1, j] + v[i, j-1] + v[i, j+1])/4.0 - v[i, j]

                    # Improve the solution by including the difference.
                    v[i, j] = v[i, j] + w*diff

                    # Store the largest absolute value of the difference
                    # as an estimate of the error.
                    err = max(err, abs(diff))

            if (show_iter):
                print('Iter =', iter, ', estimated error =', err)

            # If the error is larger than epsilon, iterate the solution again
            # Stop if the maximum number of allowed iterations is exceeded.
            if (err < eps): return (iter, v)
        
    else:
    
        newv = np.zeros((nd1, nd2, max_iter), dtype=np.float64)
        newv[:, :, 0] = v
            
        for iter in range(1, max_iter):

            err = 0.0
            newv[:, :, iter] = newv[:, :, iter-1] 
        
            # we advance by rows in the stencil and it gives the proper formula for SOR
            for i in range(1, nd1 - 1):
                for j in range(1, nd2 - 1):
                    # Calculate the difference of V(x_i, y_j) between two iterations
                    diff = (den[i, j] + newv[i-1, j, iter] + newv[i+1, j, iter] + newv[i, j-1, iter] + newv[i, j+1, iter])/4.0 - newv[i, j, iter]

                    # Improve the solution by including the difference.
                    newv[i, j, iter] = newv[i, j, iter] + w*diff

                    # Store the largest absolute value of the difference
                    # as an estimate of the error.
                    err = max(err, abs(diff))
                    
            if (show_iter):
                print('Iter =', iter, ', estimated error =', err)

            # If the error is larger than epsilon, iterate the solution again
            # Stop if the maximum number of allowed iterations is exceeded.
            if (err < eps): return (iter, newv)       

    print('Maximum number of iterations exceeded')
