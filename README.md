# pySurvONS

Librería que implementa el método de análisis de supervivencia SurvONS. Se trabajó con las librerías numpy (``np``), cvxpy (``cp``), matlplotlib (``plt``) y pandas (``pd``).

## Consideraciones

**Este modelo es altamente experimental y no se puede trabajar con cualquier dataset, debido a las operaciones matemáticas que hace el modelo por detrás. Para aumentar la probabilidad de éxito, los valores del dataset deben ser numéricos y su magnitud no debe exceder el valor 100.**

## Cómo utilizar

Para utilizar SurvONS, se debe inicializar de la siguiente forma

```py
surv = SurvONS()
surv.train(x, t0, tf, censored)
```

donde ``x`` es un DataFrame con los vectores de características de cada uno de los individuos, ``t0`` es el vector de tiempos iniciales, ``tf`` es el vector de tiempos finales y ``censored`` corresponde al vector de booleanos que indica si un individuo fue censurado.

Una vez hecho esto, el modelo está listo para ser utilizado.

## Métodos

``train(x: pd.DataFrame, t0: np.ndarray, tf: np.ndarray, censored: np.ndarray[bool], diam: float = 1)``: Entrena el modelo en base a los parámetros entregados.

```py
# x: Dataframe de pandas
# t0: np.array de numpy del tipo [1, 123, 0, 0, 2, ...]
# tf: np.array de numpy del tipo [2, 200, 43, 1000, 74, ...] 
# Se asume que tf[i] > t0[i] para todo i
# censored: np.array de numpy del tipo [True, False, True, False, False ...]

surv = SurvONS()
surv.train(x, t0, tf, censored)
```

``iterative_train(x: pd.DataFrame, t0: np.ndarray, tf: np.ndarray, censored: np.ndarray[bool])``: Optimiza iterativamente los parámetros del modelo. No funciona si el modelo no ha sido entrenado. _Advertencia: puede tomar mucho tiempo en ejecutarse._

```py
# x: Dataframe de pandas
# t0: np.array de numpy del tipo [1, 123, 0, 0, 2, ...]
# tf: np.array de numpy del tipo [2, 200, 43, 1000, 74, ...] 
# Se asume que tf[i] > t0[i] para todo i
# censored: np.array de numpy del tipo [True, False, True, False, False, ...]

surv = SurvONS()
# Requiere que el modelo ya haya sido entrenado
surv.train(x, t0, tf, censored)

surv.iterative_train(x, t0, tf, censored)
```

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

```py
# Ejemplo de uso
# x: DataFrame con las características de los individuos
X = x.to_numpy()

# Se escogen 100 individuos aleatorios y se grafican entre las iteraciones 0 y 5000
np.random.shuffle(X)
surv.plot(X[0:100], 0, 5000)

# Gráfico de un solo individuo entre las iteraciones 600 y 2000
surv.plot(X[256], 600, 2000)
```

``score(events: np.ndarray, X: list[np.ndarray[float]], cens: np.ndarray[bool])``: Calcula el concordance index del modelo con los valores entregados. ``events`` es un arreglo que contiene los tiempos reales en los que ocurren los eventos, ``X`` es la matriz de características de los individuos que se usan para calcular el score, y ``cens`` es un arreglo que indica si el individuo fue censurado o no.

```py
# Cálculo del concordance index del modelo
# x: Dataframe de pandas con las características de los individuos
# tf: Arreglo de numpy con las iteraciones en la que los individuos experimentan el evento de la forma [2, 200, 43, 1000, 74, ...] 
# cens: Arreglo de numpy que indica que individuos están censurados de booleanos de la forma [True, False, True, False, False ...]
surv.score(tf, x.to_numpy(), cens)
```
## Testing

Todos los métodos fueron probados usando unittest sobre el dataset GBSG2 de la librería sksurv. Estos tests se pueden encontrar en el archivo tests.py, y se pueden ejecutar con el comando:

```python -m unittest library/pySurvONS/tests.py```

## Agradecimientos

Este trabajo fue posible gracias a la Doctora Camila Fernandez, quien diseñó y realizó la primera implementación de este algoritmo.

Esta implementación fue realizada por Esperanza Díaz, Diego García y Paloma Silva para el curso Proyecto de Software primavera 2024.
