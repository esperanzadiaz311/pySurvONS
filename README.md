# pySurvONS
Librería que implementa el método de análisis de supervivencia SurvONS.

## Cómo utilizar

Actualmente contamos con un ejemplo de un dataset de cáncer, obtenido de scikit-survival. Para poder ejecutar dicho ejemplo se debe escribir por terminal ```python tests.py``` o ```python3 tests.py```.

En el caso donde se quiera utilizar su propio dataset, se debe inicializar de la siguiente forma

```py
surv = SurvONS()
surv.train(x, t0, tf, censored)
```
donde ``x`` es un DataFrame con los vectores de características de cada uno de los individuos, ``t0`` es el vector de tiempos iniciales, ``tf`` es el vector de tiempos finales y ``censored`` corresponde al vector de booleanos que indica si un individuo fue censurado.

Una vez hecho esto, el modelo está listo para ser utilizado.

## Métodos

```train(x, t0, tf, censored)```: Entrena el modelo en base a los parámetros entregados. 

```predict(x: int, t: int, t0: int = 0)```: Entrega la probabilidad de que el individuo con características ``x`` sobreviva hasta el instante de tiempo ``t``. No funciona si el modelo no ha sido entrenado. Opcionalmente, se le puede dar un tiempo inicial en ```t0```.

```plot(indivs: list[np.ndarray[float]] | np.ndarray[float], t0: int, tf: int)```: Genera un gráfico de la probabilidad de superivencia de un grupo de individuos en el intervalo de tiempo entre ``t0`` y ``tf``. ``indivs`` es una lista de vectores de característica de los individuos que se quieren graficar, o es un vector de características en caso de que solo se quiera graficar la probabilidad de supervivencia de un individuo. No funciona si el modelo no ha sido entrenado.
