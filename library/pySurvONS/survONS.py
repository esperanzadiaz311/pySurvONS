import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import instgrad, generalized_projection
from lifelines.utils import concordance_index

# Implementación en Python de SurvONS

class SurvONS():
       
    def __init__(self) -> None:
        self.trained = False
        self.beta = np.zeros((1,))

    
    # X: vectores de características de cada uno de los individuos
    # t0: vector de tiempos iniciales
    # tf: vector de tiempos finales
    # censored: vector de booleanos que indica si un individuo fue censurado
    # D: Diámetro del espacio de las características
    # gamma: Vector de valores de learning rate
    # n: Número de iteraciones
    # epsilon: Vector de valores de epsilon (epsilon_k = 1/(gamma_k * D) ^2)
    # R: Matriz de los factores de riesgo de los individuos
    # max0: Setea el valor de gamma_temp a 0
    def __surv_ons(self, X: np.ndarray, t0: np.ndarray, tf: np.ndarray,
                   censored: np.ndarray[bool], D: float, gamma: np.ndarray[float], n: int, 
                   epsilon: np.ndarray[float], R: np.ndarray[np.ndarray[int]], max0: bool = False) -> dict:
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

        beta_boa_arr = np.zeros((n, d))
        grad_boa = np.zeros((n, d))
        grad_boa_hat = np.zeros((n, d, K))
        pi_boa_arr = np.zeros((n, K))
        lik_boa = np.zeros((n, 1))
        gamma_temp = np.zeros((n, 1))

        for t in range(1, n): # la iteración 0 da todo 0 => mata todo
            #print(f"iteracion {t}")
            beta_boa = np.matmul(beta, pi_boa)
            grad_boa[t], hess_boa , lik_boa[t] = instgrad(t, t0, tf, censored, X, beta_boa, R[t])

            norm_grad_boa = np.linalg.norm(grad_boa[t])
            algo = np.matmul(np.transpose(grad_boa[t]), hess_boa)
            mu = np.matmul(algo, grad_boa[t]) / max(1e-9, norm_grad_boa**4)

            gamma_t = 2*((-1/max(1e-9, mu))*np.log(1 + mu*norm_grad_boa*D) + norm_grad_boa*D) / (max(1e-9, norm_grad_boa * D) ** 2)
            if (max0):
                gamma_t = 0
            gamma_temp[t] = gamma_t

            for i in range(0, K):
                beta_arr[t, :, i] = beta[:, i]
                
                gamma_max = max(gamma_t/4, gamma[i])

                grad_hat = grad_boa[t] * (1 + gamma_max * np.matmul(np.transpose(grad_boa[t]), beta[:, i]- beta_boa))
                grad_boa_hat[t,:,i] = grad_hat
                
                a_inv  = np.full((d,d), a_inv_arr[:,:,i])
                temp = a_inv @ grad_hat
                temp2 = np.outer(temp, temp.T) # es una matriz simétrica -> (1,3) == (3,1) y su diagonal con valores distintos

                a_inv = a_inv - (temp2/(1+ np.matmul(grad_hat.T, temp)))
                a_inv_arr[:,:,i] = a_inv
                
                beta[:,i] -= gamma[i]**-1 * np.matmul(a_inv, grad_hat)


                if (np.sqrt(np.matmul(np.transpose(beta[:,i]), beta[:,i])) > D):
                    beta[:, i] = generalized_projection(a_inv, beta[:, i], D, d)

            term1 = (gamma * (np.matmul(np.transpose(grad_boa[t]), beta-np.matmul(beta_boa, np.transpose(np.ones((K, 1)))))))[..., np.newaxis]

            pi_boa2 = np.exp(np.maximum(-100, np.minimum(100, np.log(pi_boa2) - term1 - term1**2)))
            pi_boa2 /= np.sum(pi_boa2)

            gamma_dot_pb2 = np.array([gamma]).T * pi_boa2
            pi_boa = gamma_dot_pb2 / np.sum(gamma_dot_pb2)


            if(t < n):
                beta_boa_arr[t] = (np.matmul(beta, pi_boa)).flatten()
                pi_boa_arr[t] = pi_boa.flatten()
        
        return {"beta_arr": beta_arr, "beta_boa_arr": beta_boa_arr, "pi_boa_arr": pi_boa_arr, 
                "lik_boa": lik_boa, "gamma_temp": gamma_temp, "grad_boa": grad_boa, 
                "grad_boa_hat": grad_boa_hat}
    
    # Riesgo instantáneo de un individuo en un instante t
    # xi: Vector de caraterísticas del individuo
    # t0: Tiempo inicial del individuo
    # t: Instante que se quiere predecir
    def __hazard(self, xi: np.ndarray, t0: int, t: int) -> float:
        return np.exp(np.matmul(self.beta.T, xi)) * (t >= t0)

    # Probabilidad de que un individuo sobreviva hasta
    # un instante dado
    # xi: Vector de caraterísticas del individuo
    # t0: Tiempo inicial del individuo
    # t: Instante que se quiere predecir
    def __survive(self, xi: np.ndarray, t0: int, t: int) -> float:
        if (t < t0):
            return 1
        return np.exp(-1 * np.exp(np.matmul(self.beta.T, xi)) * (t - t0))

    # Entrena el modelo de SurvONS a partir de un dataset
    def train(self, x: pd.DataFrame, t0: np.ndarray, tf: np.ndarray, censored: np.ndarray[bool]) -> None:

        X = x.to_numpy()

        N = X.shape[0] # Número de individuos
        n_it = int(max(tf))
        self.t_max = n_it
        d = X.shape[1]

        gamma = np.arange(np.log(1/n_it), np.log(5*d), 1.2)
        gamma = np.exp(gamma)

        D = 2.5
        epsilon = 1/(gamma*D) ** 2
        R = [[] for _ in range(n_it)]
        for t in range(0, n_it):
            R[t].append(0)
        for i in range(1, N):
            t1 = max(1,int(np.floor(t0[i])-1)) # tiempo en el que i entra al estudio
            t2 = min(n_it,int(np.floor(tf[i])+1)) # tiempo en el que i sale del estudio
            for t in range(t1, t2):
                R[t].append(i)

        survons = self.__surv_ons(X, t0, tf, censored, D, gamma, n_it, epsilon, R)
        self.beta = survons["beta_boa_arr"][n_it-1,:]
        self.trained = True

    # Probabilidad de supervivencia de un individuo
    # en un tiempo dado
    # i: individuo a predecir
    # t: tiempo en el que se quiere ver la 
    #    probabilidad de supervivencia
    def predict(self, x: np.ndarray[float], t: int, t0: int = 0) -> float:
        if not self.trained:
            print("Train the model before doing predictions")
            return
        return self.__survive(x, t0, t)

    def predict_time(self, x: np.ndarray[float], t0: int = 0) -> float:
        time = 0
        for t in range(t0, self.t_max + 1):
            time += t*self.__hazard(x, t0, t)

        return time


    # Grafica la probabilidad de supervivencia de un grupo de
    # individuos en un intervalo de tiempo
    # indivs: matriz de características de los individuos
    #         a graficar o una lista si es solo
    #         un individuo
    # t0: tiempo inicial
    # tf: tiempo final
    def plot(self, indivs: list[np.ndarray[float]] | np.ndarray[float], t0: int, tf: int) -> None:
        if not self.trained:
            print("Train the model before doing predictions")
            return
        
        if len(indivs) == 0:
            return
        
        survival = []
       
        if isinstance(indivs[0], (np.floating, float)):
            survival = [self.predict(indivs, t) for t in range(t0, tf+1)]
            plt.plot([i for i in range(t0, tf+1)], survival)
        
        else:
            survival = [[self.predict(xi, t) for t in range(t0, tf+1)] for xi in indivs]
            colors = plt.cm.jet(np.linspace(0, 1, len(survival)))
            for indv in survival:
                plt.plot([i for i in range(t0, tf+1)], indv, label=f"Individuo {survival.index(indv) + 1}", color=colors[survival.index(indv)])
        
        plt.title("Tiempo v/s Probabilidad de Supervivencia")
        plt.xlabel("Tiempo")
        plt.ylabel("Probabilidad de Supervivencia")
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    # Cálculo de concordance index
    def score(self, events, X, cens) -> float:
        
        preds = [self.predict_time(X[i]) for i in range(len(events))]

        return concordance_index(events, preds, cens)