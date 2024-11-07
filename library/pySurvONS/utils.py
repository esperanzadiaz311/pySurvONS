import numpy as np
import cvxpy as cp
import pandas as pd

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
        xi = X[i]
        lik = lik - ((t-1 < u[i]) * (u[i] <= t) * delta[i] * np.matmul(np.transpose(beta), X[i])[0] + np.exp(np.matmul(np.transpose(beta), X[i]))[0] * max(0, min(t,u[i]) - max(t0[i],t-1))) / N
        hess = hess + np.outer(xi, np.transpose(xi)) * np.exp(np.dot(beta.flatten(), xi)) * max(0, min(t, u[i]) - max(t0[i], t-1)) / N
        grad = grad + (xi * np.exp(np.matmul(np.transpose(beta), X[i])[0]) * max(0, min(t,u[i]) - max(t0[i], t-1)) / N)[:,np.newaxis]
        
        if (t-1 < u[i] and u[i] <= t and delta[i]):
            grad = grad - (xi / N)[:,np.newaxis]
    return grad.flatten(), hess, lik


def generalized_projection(P, theta, D, d):
    x = cp.Variable(d) # dimensión (d,)
    #P_inverted = np.linalg.inv(P)
    matrix_x_theta = cp.matrix_frac(x-theta, P) # min (x-theta) P^-1 (x-theta)
    #matrix_x_theta = cp.quad_form(x-theta, P_inverted)
    objective = cp.Minimize(matrix_x_theta)

    constraint = [cp.norm(x,2) <= D]
    problem = cp.Problem(objective, constraint)
    
    problem.solve()
    status = problem.status
    #print("status", status)

    if (status != cp.OPTIMAL):
        problem.solve(solver = cp.SCS)
    #print("status", status)

    if (status != cp.OPTIMAL):
        print(f"Generalized projection error: optimization is not optimal and ends with status {status}", x.value)
    #print("x.value", x.value)
    return x.value

def date_discretization(dates, depth="day") -> np.ndarray[int]:
    
    dates_to_datetime = pd.to_datetime(dates)
    depth_map = {
        "day": dates_to_datetime.date,  # Fecha completa (sin hora)
        "month": dates_to_datetime.to_period('M'),  # Solo mes y año
        "year": dates_to_datetime.year,  # Solo el año
    }

    if depth not in depth_map:
        raise ValueError(f"Discretización no encontrada. Solo se soporta 'day', 'month' y 'year'")

    # Ordenar las fechas sin repetir
    sorted_dates = np.unique(depth_map[depth])

    # Crear un índice secuencial, empezando desde 0
    date_to_discrete = {date: i for i, date in enumerate(sorted_dates)}

    # Aplicar el mapeo a la lista original de fecha
    discretized_dates = np.ndarray((len(dates), ), dtype=int)
    
    for i in range(len(dates_to_datetime)):
        discretized_dates[i] = date_to_discrete[depth_map[depth][i]]

    return discretized_dates

def get_censored_values(values, max_value) -> np.ndarray[bool]:
    
    cens = np.ndarray((len(values),), dtype=bool)

    for i in range(len(values)):
        leq = (values[i] < max_value)
        cens[i] = leq

    return cens