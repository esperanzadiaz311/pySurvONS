# pySurvONS

Librería que implementa el método de análisis de supervivencia SurvONS.

## Cómo utilizar

Actualmente contamos con un ejemplo de un dataset de cáncer, obtenido de scikit-survival. Para poder ejecutar dicho ejemplo se debe escribir por terminal ``python tests.py`` o ``python3 tests.py``.

En el caso donde se quiera utilizar su propio dataset, se debe inicializar de la siguiente forma

```py
surv = SurvONS()
surv.train(x, t0, tf, censored)
```

donde ``x`` es un DataFrame con los vectores de características de cada uno de los individuos, ``t0`` es el vector de tiempos iniciales, ``tf`` es el vector de tiempos finales y ``censored`` corresponde al vector de booleanos que indica si un individuo fue censurado.

Una vez hecho esto, el modelo está listo para ser utilizado.

## Métodos

``train(x: pd.DataFrame, t0: np.ndarray, tf: np.ndarray, censored: np.ndarray[bool], diam: float = 1)``: Entrena el modelo en base a los parámetros entregados.

``iterative_train(x: pd.DataFrame, t0: np.ndarray, tf: np.ndarray, censored: np.ndarray[bool])``: Optimiza iterativamente los parámetros del modelo. No funciona si el modelo no ha sido entrenado. _Advertencia: puede tomar mucho tiempo en ejecutarse._

``predict(indivs: list[np.ndarray[float]] | np.ndarray[float], t: int, t0: np.ndarray[float] | int = 0)``: Si ``indivs`` es solo un individuo, entrega la probabilidad de que el individuo sobreviva hasta el instante de tiempo ``t``. Si ``indivs`` es un listado de individuos, entrega la probabilidad de que cada individuo sobreviva hasta el tiempo ``t``. No funciona si el modelo no ha sido entrenado. Opcionalmente, se le puede dar un tiempo inicial ``t0`` o puede recibir una lista de tiempos iniciales si se entrega más de un individuo.

```py
# Ejemplo de uso
# x: DataFrame con las características de los individuos
indivs = x.to_numpy()[0:50]

# Predicción de un solo individuo
surv.predict(indivs[1], 2000, df["initial_time"].to_numpy()[1])

# Predicción de múltiples individuos con tiempos iniciales distintos
surv.predict(indivs, 2000, df["initial_time"].to_numpy()[0:50])

# Predicción de múltiples individuos con tiempos iniciales iguales (t0 = 0)
surv.predict(indivs, 2000)
```

``predict_time(indivs: list[np.ndarray[float]] | np.ndarray[float], t0: np.ndarray[float] | int = 0)``: Si ``indivs`` son las características de un individuo, entrega el intervalo de tiempo en el que se estima experimente el evento, si es un arreglo de características de múltiples individuos, entrega un vector con el tiempo estimado para cada uno de los individuos. Puede recibir un tiempo inicial en ``t0``, o puede recibir una lista de tiempos iniciales si se entrega más de un individuo.

```py
# Ejemplo de uso
# x: DataFrame con las características de los individuos
indivs = x.to_numpy()[0:50]

# Predicción de un solo individuo
surv.predict_time(indivs[1], df["initial_time"].to_numpy()[1])

# Predicción de múltiples individuos con tiempos iniciales distintos
surv.predict_time(indivs, df["initial_time"].to_numpy()[0:50])

# Predicción de múltiples individuos con tiempos iniciales iguales (t0 = 0)
surv.predict_time(indivs)
```

``plot(indivs: list[np.ndarray[float]] | np.ndarray[float], t0: int, tf: int)``: Genera un gráfico de la probabilidad de superivencia de un grupo de individuos en el intervalo de tiempo entre ``t0`` y ``tf``. ``indivs`` es una lista de vectores de característica de los individuos que se quieren graficar, o es un vector de características en caso de que solo se quiera graficar la probabilidad de supervivencia de un individuo. No funciona si el modelo no ha sido entrenado.

``score(events: np.ndarray, X: list[np.ndarray[float]], cens: np.ndarray[bool])``: Calcula el concordance index del modelo con los valores entregados. ``events`` es un arreglo que contiene los tiempos reales en los que ocurren los eventos, ``X`` es la matriz de características de los individuos que se usan para calcular el score, y ``cens`` es un arreglo que indica si el individuo fue censurado o no.
