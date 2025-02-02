import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import instgrad, generalized_projection
from lifelines.utils import concordance_index
from cvxpy import DCPError, SolverError

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
                #print("beta_boa_arr[t]:", beta_boa_arr[t])
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
    
    # Verifica si un modelo fue entrenado
    def __check_trained(self):
        if not self.trained:
            print("Entrene el modelo antes de hacer predicciones.")
            return False
        return True

    # Entrena el modelo de SurvONS a partir de un dataset
    # x: Dataset con las características de cada individuo
    # t0: Arreglo de tiempos iniciales de cada individuo
    # tf: Arreglo de tiempos finales de cada individuo 
    # censored: Vector de booleanos que indica si un individuo fue censurado
    # diam: Diametro máximo del espacio de características del dataset
    def train(self, x: pd.DataFrame, t0: np.ndarray, tf: np.ndarray, censored: np.ndarray[bool], diam: float = 1) -> None:
        
        X = x.to_numpy()

        N = X.shape[0] # Número de individuos
        n_it = int(max(tf))
        self.t_max = n_it
        d = X.shape[1]

        gamma = np.arange(np.log(1/n_it), np.log(5*d), 1.2)
        gamma = np.exp(gamma)

        D = diam
        epsilon = 1/(gamma*D) ** 2
        R = [[] for _ in range(n_it)]
        for t in range(0, n_it):
            R[t].append(0)
        for i in range(1, N):
            t1 = max(1,int(np.floor(t0[i])-1)) # tiempo en el que i entra al estudio
            t2 = min(n_it,int(np.floor(tf[i])+1)) # tiempo en el que i sale del estudio
            for t in range(t1, t2):
                R[t].append(i)
        count = 0
        while count < 10 and not self.trained:
            try:
                survons = self.__surv_ons(X, t0, tf, censored, D, gamma, n_it, epsilon, R)
            except (DCPError, SolverError) as e:
                D *= 2
                epsilon = 1/(gamma*D) ** 2
                print(f"Valor de D muy pequeño. Probando con D={D}")
                count += 1
            else:
                self.beta = survons["beta_boa_arr"][n_it-1,:]
                self.trained = True
                self.D = D
                print("Entrenamiento exitoso, para optimizar el modelo utilizar iterative_train")
            
        if count >= 10:
            print("No se pudieron encontrar parámetros adecuados para el dataset entregado")
    
    # Optimiza los parámetros de un modelo ya entrenado
    # x: Dataset con las características de cada individuo
    # t0: Arreglo de tiempos iniciales de cada individuo
    # tf: Arreglo de tiempos finales de cada individuo 
    # censored: Vector de booleanos que indica si un individuo fue censurado
    def iterative_train(self, x: pd.DataFrame, t0: np.ndarray, tf: np.ndarray, censored: np.ndarray[bool]) -> None:

        if not self.__check_trained():
            return

        factors = [0.8, 0.9, 0.95]

        X = x.to_numpy()

        base_concordance = self.score(tf, X, censored)

        N = X.shape[0] # Número de individuos
        n_it = int(max(tf))
        self.t_max = n_it
        d = X.shape[1]

        gamma = np.arange(np.log(1/n_it), np.log(5*d), 1.2)
        gamma = np.exp(gamma)

        R = [[] for _ in range(n_it)]
        for t in range(0, n_it):
            R[t].append(0)
        for i in range(1, N):
            t1 = max(1,int(np.floor(t0[i])-1)) # tiempo en el que i entra al estudio
            t2 = min(n_it,int(np.floor(tf[i])+1)) # tiempo en el que i sale del estudio
            for t in range(t1, t2):
                R[t].append(i)

        new_concordance = base_concordance
        index = 0
        while index < len(factors):
            new_D = self.D * factors[index]
            epsilon = 1/(gamma*new_D) ** 2
            print(f"Probando D={new_D}")
            try:
                survons = self.__surv_ons(X, t0, tf, censored, new_D, gamma, n_it, epsilon, R)
            except (DCPError, SolverError) as e:
                index +=1
                continue
            else:
                old_beta = self.beta
                old_D = self.D
                self.beta = survons["beta_boa_arr"][n_it-1,:]
                self.D = new_D

                new_concordance = self.score(tf, X, censored)
                print(f"new_concordance: {new_concordance}")

            if (new_concordance > base_concordance):
                base_concordance = new_concordance
                continue
            else:
                self.beta = old_beta
                self.D = old_D
                index +=1
                continue

        print(f"Valor de D final: {self.D}")
         

    # Probabilidad de supervivencia de un individuo
    # en un tiempo dado
    # indivs: individuo a predecir o listado de individuos a predecir
    # t: tiempo en el que se quiere ver la 
    #    probabilidad de supervivencia
    # t0: tiempo inicial de individuo o listado de tiempos iniciales
    #     de cada individuo 
    
    def predict(self, indivs: list[np.ndarray[float]] | np.ndarray[float], t: int, t0: np.ndarray[float] | int = 0) -> float | None:
        if not self.__check_trained():
            return

        if len(indivs) == 0:
            return
       
        if isinstance(indivs[0], (np.floating, float)):
            return self.__survive(indivs, t0, t)

        else:
            prob = []
            for i in range(len(indivs)):
                if type(t0) == int:
                    prob.append(self.__survive(indivs[i], t0, t))
                else:
                    prob.append(self.__survive(indivs[i], t0[i], t))
            return prob
        
    # Tiempo estimado en el que un individuo(s) experimentará(n) el 
    # indivs: individuo a predecir o listado de individuos a predecir
    # t0: tiempo inicial de individuo o listado de tiempos iniciales
    #     de cada individuo 
    def predict_time(self, indivs: list[np.ndarray[float]] | np.ndarray[float], t0: np.ndarray[float] | int = 0) -> float | None:
        if not self.__check_trained():
            return

        if len(indivs) == 0:
            return
       
        if isinstance(indivs[0], (np.floating, float)):
            time = 0
            probs = 0
            for t in range(t0, self.t_max + 1):
                p = self.__survive(indivs, t0, t)
                time += t*p
                probs += p

            return time/probs
        else:
            predicts = []
            for i in range(len(indivs)):
                time = 0
                probs = 0
                if type(t0) == int:
                    for t in range(t0, self.t_max + 1):
                        p = self.__survive(indivs[i], t0, t)
                        time += t*p
                        probs += p
                else:
                    for t in range(t0[i], self.t_max + 1):
                        p = self.__survive(indivs[i], t0[i], t)
                        time += t*p
                        probs += p

                predicts.append(time/probs)
            return predicts

        
    # Grafica la probabilidad de supervivencia de un grupo de
    # individuos en un intervalo de tiempo
    # indivs: matriz de características de los individuos
    #         a graficar o una lista si es solo
    #         un individuo
    # t0: tiempo inicial
    # tf: tiempo final
    def plot(self, indivs: list[np.ndarray[float]] | np.ndarray[float], t0: int, tf: int) -> None:
        if not self.__check_trained():
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
        plt.show()

    # Cálcula el concordance index del modelo con los datos entregados
    # events: arreglo con los valores reales del tiempo en el que cada individuo experimenta el evento
    # X: matriz con las características de los individuos
    # cens: vector de booleanos que indica si un individuo fue censurado
    def score(self, events: np.ndarray, X: list[np.ndarray[float]], cens: np.ndarray[bool]) -> float | None:
        if not self.__check_trained():
            return
        
        if (len(events) == 0 or len(X) == 0 or len(cens) == 0):
            return
        
        if (len(events) != len(X) or len(events) != len(cens) or len(cens) != len(X)):
            return
        
        preds = [self.predict_time(X[i]) for i in range(len(events))]

        return concordance_index(events, preds, cens)