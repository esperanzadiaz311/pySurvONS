import numpy as np
import cvxpy as cp

# Función auxiliar que retorna el Likelihood, Gradiente y Hessiano 

# t: iteración actual 
# t0: Arreglo de tiempos iniciales de los individuos
# u: Arreglo de tiempos en los que los individuos experimentan el evento/son censurados (hat_T en R)
# delta: Arreglo de booleanos que indican si un individuo experimentó el evento o fue censurado
# X: Matriz con los vectores de características de cada uno de los individuos
# beta: Vector de "pesos" de las características
# R_t: Arreglo del factor de riesgo de los individuos en esta iteración

def instgrad(t, t0, u, delta, X, beta, R_T) -> list:
    N = len(t0)
    d = np.shape(X)[1]
    
    grad = np.zeros((d, 1)) # Gradiente
    hess = np.zeros((d, d)) # Hessiano
    lik = 0 # Likelihood
    
    for i in R_T:
        xi = np.array([X[i]])
        lik -= ((t-1 < u[i]) * (u[i] <= t) * delta[i] * np.matmul(np.transpose(beta), X[i]) + np.exp(np.matmul(np.transpose(beta), X[i])) * max(0, min(t,u[i]) - max(t0[i],t-1))) / N
        hess += np.matmul(xi, np.transpose(xi))[0,0] * np.exp(np.dot(beta, xi))[0] * max(0, min(t, u[i]) - max(t0[i], t-1)) / N
        grad += (xi * np.exp(np.matmul(np.transpose(beta), X[i])) * max(0, min(t,u[i]) - max(t0[i], t-1)) / N).T

        if (t-1 < u[i] and u[i] <= t and delta[i]):
            grad -= xi / N

    return grad.flatten(), hess, lik


def generalized_projection(P, theta, D, d):
    x = cp.Variable(d) # dimensión (d,)

    matrix_x_theta = cp.matrix_frac(x-theta, P)
    objective = cp.Minimize(matrix_x_theta)

    constraint = [cp.norm2(x) <= D]
    problem = cp.Problem(objective, constraint)
    
    status = problem.status
    print("status", status)

    if (status != cp.OPTIMAL):
        problem.solve(solver = cp.SCS)

    if (status != cp.OPTIMAL):
        #print(f"Generalized projection error: optimization is not optimal and ends with status {status}", x.value)
        pass
    print("x.value", x.value)
    return x.value
