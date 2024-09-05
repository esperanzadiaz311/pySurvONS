import numpy as np
from utils import instgrad, generalized_projection

# Implementación en Python de SurvONS

# t0: Arreglo de tiempos iniciales de los individuos
# u: Arreglo de tiempos en los que los individuos experimentan el evento/son censurados (hat_T en R)
# delta: Arreglo de booleanos que indican si un individuo experimentó el evento o fue censurado
# X: Matriz con los vectores de características de cada uno de los individuos
# D: Diametro del espacio de las características
# gamma: Vector de valores de learning rate
# n: Número de iteraciones
# epsilon: Vector de valores de epsilon (epsilon_k = 1/(gamma_k * D) ^2)
# R: Matriz de los factores de riesgo de los individuos
# max0: Setea el valor de gamma_temp a 0

def surv_ons(t0, u, delta, X, D, gamma, n, epsilon, R, max0 = False):
    N = len(t0)
    d = np.shape(X)[1]  # entrega la segunda dimensión de X (número de características)
    K = len(gamma)

    beta = np.zeros((d, K))
    beta_arr = np.zeros((n, d, K))
    a_inv_arr = np.zeros((d, d, K))

    for i in range(0, K):
        a_inv_arr[:, :, i] = (1/epsilon[i]) * np.eye(d)

    pi_boa = np.full((K, 1), 1/K)
    pi_boa2 = pi_boa
    L = np.zeros((K,1))
    renorm = np.zeros((K, 1))

    beta_boa_arr = np.zeros((n, d))
    grad_boa = np.zeros((n, d))
    grad_boa_hat = np.zeros((n, d, K))
    pi_boa_arr = np.zeros((n, K))
    lik_boa = np.zeros((n, 1))
    gamma_temp = np.zeros((n, 1))

    for t in range(0, n):

        beta_boa = np.matmul(beta, pi_boa)

        grad_boa[t], hess_boa , lik_boa[t] = instgrad(t, t0, u, delta, X, beta_boa, R[t])

        norm_grad_boa = np.linalg.norm(grad_boa) 
        mu = (np.matmul(np.matmul(np.transpose(grad_boa), hess_boa), grad_boa) / max(1e-9, norm_grad_boa**4)).astype(np.float64)

        gamma_t = np.float64(2*((-1/mu)*np.log(1 + mu*norm_grad_boa*D) + norm_grad_boa*D) / max(1e-9, norm_grad_boa * D) ** 2)
        if (max0):
            gamma_t = 0
        gamma_temp[t] = gamma_t

        for i in range(0, K):
            beta_arr[t, :, i] = beta[:, i]
            gamma_max = max(gamma_t/4, gamma[i])

            grad_hat = np.dot(grad_boa, (1 + gamma_max * np.cross(grad_boa, beta[:, i]- beta_boa))).astype(np.float64)
            grad_boa_hat[t,:,i] = grad_hat

            a_inv  = np.full((d,d), a_inv_arr[:,:,i])
            temp = np.matmul(a_inv, grad_hat)
            a_inv -= np.cross(temp, temp.T)/(1+ np.cross(grad_hat, temp)[0])
            a_inv_arr[:,:,i] = a_inv
            beta[:,i] -= gamma[i]**-1 * np.matmul(a_inv, grad_hat)

            if (np.sqrt(np.cross(np.transpose(beta[:,i]), beta[:,i])[0]) > D):
                beta[:, i] = generalized_projection(a_inv, beta[:, i], D)

        term1 = np.dot(gamma, (np.cross(grad_boa, beta-np.matmul(beta_boa, np.transpose(np.ones((K, 1)))))))
        
        # revisar si es necesario maximum y minimum, o si sirve max y min
        pi_boa2 = np.exp(np.maximum(-100, np.minimum(100, np.log(pi_boa2) - term1 - term1**2)))
        pi_boa2 /= np.sum(pi_boa2)

        gamma_dot_pb2 = np.dot(gamma * pi_boa2)
        pi_boa = gamma_dot_pb2 / sum(gamma_dot_pb2)

        if(t < n):
            beta_boa_arr[t+1] = np.matmul(beta, pi_boa)
            pi_boa_arr[t+1] = pi_boa
    
    return [beta_arr, beta_boa_arr, pi_boa_arr, 
            lik_boa, gamma_temp, grad_boa, grad_boa_hat]