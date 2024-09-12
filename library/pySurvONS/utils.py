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

    print("beta (instgrad)", beta)
    print("R_T:", R_T)
    print("t:", t)
    for i in R_T:
        print("HOLA soy el for")
        xi = np.array([X[i]])
        print("t0[i]: ", t0[i])
        print("u[i]: ", u[i])
        print("min(t,u[i]): ", min(t,u[i]))
        print("max(t0[i],t-1): ", max(t0[i],t-1))
        print("max(0, min(t,u[i]) - max(t0[i],t-1)): ", max(0, min(t,u[i]) - max(t0[i],t-1)))
        lik = lik - ((t-1 < u[i]) * (u[i] <= t) * delta[i] * np.matmul(np.transpose(beta), X[i])[0] + np.exp(np.matmul(np.transpose(beta), X[i]))[0] * max(0, min(t,u[i]) - max(t0[i],t-1))) / N
        print("lik", lik)
        hess = hess + np.matmul(xi, np.transpose(xi)) * np.exp(np.dot(beta, xi))[0] * max(0, min(t, u[i]) - max(t0[i], t-1)) / N
        print("hess", hess)
        grad = grad + (xi * np.exp(np.matmul(np.transpose(beta), X[i])[0]) * max(0, min(t,u[i]) - max(t0[i], t-1)) / N).T
        print("grad", grad)

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
    print("status", status)

    if (status != cp.OPTIMAL):
        print(f"Generalized projection error: optimization is not optimal and ends with status {status}", x.value)
        pass
    print("x.value", x.value)
    return x.value
