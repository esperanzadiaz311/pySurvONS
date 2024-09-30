# pySurvONS
Librería que implementa el método de análisis de supervivencia SurvONS.

## Cómo utilizar

Actualmente contamos con un ejemplo de un dataset de cáncer, obtenido de scikit-survival. Para poder ejecutar dicho ejemplo se debe escribir por terminal ```python tests.py``` o ```python3 tests.py```.

En el caso donde se quiera utilizar su propio dataset, se debe inicializar de la siguiente forma

```py
surv = SurvONS(x, t0, tf, censored)
surv.train()
```
donde ``x`` es un DataFrame con los vectores de características de cada uno de los individuos, ``t0`` es el vector de tiempos iniciales, ``tf`` es el vector de tiempos finales y ``censored`` corresponde al vector de booleanos que indica si un individuo fue censurado.

Una vez hecho esto, el modelo está listo para ser utilizado.

## Métodos

```train()```: Entrena el modelo en base a los parámetros entregados durante la inicialización del modelo.

```predict(i: int, t: int)```: Entrega la probabilidad de que el individuo ``i`` del dataset entregado sobreviva hasta el instante de tiempo ``t``. No funciona si el modelo no ha sido entrenado.

```plot(indivs: array[int], t0: int, tf: int)```: Genera un gráfico de la probabilidad de superivencia de los individuos del arreglo ``indivs`` en el intervalo de tiempo entre ``t0`` y ``tf``. No funciona si el modelo no ha sido entrenado.
