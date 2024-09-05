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
    lik = np.zeros((1,1)) # Likelihood
    
    for i in R_T:
        lik -= ((t-1 < u[i]) * (u[i] <= t) * delta[i] * np.matmul(np.transpose(beta), X[i]) + np.exp(np.matmul(np.transpose(beta), X[i])) * max(0, min(t,u[i]) - max(t0[i],t-1))) / N
        hess += np.matmul(X[i], np.transpose(X[i])) * np.exp(np.dot(beta, X[i]))[0] * max(0, min(t, u[i]) - max(t0[i], t-1)) / N
        grad += X[i] * np.exp(np.cross(beta, X[i])[0]) * max(0, min(t,u[i]) - max(t0[i], t-1)) / N

    if (t-1 < u[i] and u[i] <= t and delta[i]):
        grad -= X[i] / N

    return grad, hess, lik


def generalized_projection(P, theta, D, d=4):
    x = cp.Variable(d)
    objective = cp.Minimize(cp.MatrixFrac(x-theta, P))
    constraint = [cp.norm2(x) <= D]
    problem = cp.Problem(objective, constraint)
    problem.solve()
    status = problem.status

    if (status != 'optimal'):
        problem.solve(solver = cp.SCS)

    if (status != 'optimal'):
        print(f"Generalized projection error: optimization is not optimal and ends with status {status}")

    return x.value
