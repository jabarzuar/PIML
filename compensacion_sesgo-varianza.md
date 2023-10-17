 
 # Compensación de varianza y sesgo


Todo modelo predictivo está sujeto a algún tipo de error, que imposibilita obtener error nulo tanto en los sets de training como de testeo. Dependiendo de cómo esté construido el modelo, este puede ser propenso a cometer distintos tipos de error. En este caso, analizaremos el error por exceso de varianza y exceso de sesgo, y cómo buscar el mejor equilibrio entre ellos para obtener modelos más precisos. 

Para poder discutir estos conceptos, es útil entender lo que es un **estimador**. De acuerdo con Goodfellow et al, _point estimation_ es “el intento de obtener la ‘mejor’ predicción de alguna cantidad de interés” (2016), de manera que un buen estimador será una función que entregue un valor estimado cercano al real. Por ejemplo, dado un grupo de **$M$** personas, se puede estimar la altura media **$h$** de este grupo tomando una muestra de menor tamaño **$m$** y calculando su altura media **$\hat{h}_m$**. El valor esperado de la función utilizada para encontrar el estimador puede o no ser igual al valor real del parametro que se está estimando; en caso de que ambos sean iguales, se dice que el estimador no tiene sesgo.  El **sesgo** o _bias_ de un estimador, entonces, se refiere a la diferencia entre el valor esperado del mismo y el valor real que se está estimando. En el ejemplo anterior, esto se sería **$bias(\hat{h}_m) = E(\hat{h}_m) - h$**. Un estimador se denomina no sesgado si su sesgo es igual a cero y sesgado si no lo es; también se puede decir que un estimador es asintóticamente no sesgado si su sesgo disminuye a medida que aumenta la cantidad de datos, de manera que **$\displaystyle \lim_{m \to \infty}bias(\hat{h}_m) = 0$**.

Siguiendo con nuestro ejemplo del estimador, si repitieramos el experimento con una nueva muestra de personas, es intuitivo que el promedio de sus estaturas será distinto al primer valor obtenido, a pesar de que estas personas pertenecen a la misma población. De hecho, lo esperable es que cada vez que se resamplee el valor del estimador obtenido sea un poco distinto; este cambio es un reflejo de la varianza del estimador. La **varianza** de los estimadores es una propiedad muy importante, que nos indica cómo esperamos que varíe el estimador al resamplear de forma independiente. Dada una muestra de datos de tamaño **$m$**, su varianza se calcula como $Var(f(x))=\frac{\Sigma (x_i - \mu)^2}{m-1}=E(X^2)-E(X)^2$, en que **$\mu$** es el promedio de la población. La varianza es tan importante porque una varianza alta implica que pequeños cambios en el input de un modelo o función pueden significar cambios muy grandes en su output, lo que generalmente no es deseable, además de que esta función será mucho más vulnerable frente a _outliers_ y datos erroneos.

En el contexto de _machine learning_, la varianza y el sesgo de un modelo están relacionadas mediante el concepto de **capacidad**. Goodfellow et al definen capacidad como "la habiidad de un modelo de ajustarse a una amplia variedad de funciones" (2016), lo que en un principio puede parecer ambiguo. Pongamos un nuevo ejemplo: digamos que se tiene una serie de puntos pertenecientes a una curva, por ejemplo una parábola, a los que se les agrega ruido, y que contamos con un modelo cuyo fin es encontrar la curva que mejor se adapta a estos puntos (esto segun algún criterio en específico, como puede ser la que tenga un menor error cuadrático medio). Si nuestro modelo sólo puede entregar como solución funciones lineales se dice que tiene poca capacidad, y obviamente las soluciones que entregue no se asemejarán a la parábola que generó los puntos. Sin embargo, si tomo nuevos puntos generados a partir de la misma parábola, el modelo entregará una recta similar a la primera. En otras palabras, nuestro modelo tiene mucho sesgo, pero su varianza es baja. Ahora supongamos que el modelo puede entregar polinomios de grado muy alto como solución, diremos que tiene mucha capacidad, y probablemente entregue una curva de un grado tan grande que la curva pase casi exactamente por cada punto (recordemos que los puntos tienen ruido, por lo que se alejan un poco de la parábola), luego si resampleamos, el modelo encontrará otra curva que se ajuste muy bien a los nuevos puntos, pero ambas curvas serán muy distintas entre sí. En otras palabras, el modelo tiene poco sesgo pero mucha varianza.


| ![1_9hPX9pAO3jqLrzt0IE3JzA](https://github.com/jabarzuar/PIML/assets/101306821/e0093cbb-386b-40b5-b154-88c0a4c87cf9) |
|:---|
| Figura 1: Se muestra un ejemplo de lo descrito en el párrafo anterior. A la izquierda se encuentra el resultado que entregaría un mdelo con demasiada capacidad, que tendría poco sesgo, ya que la curva se ajusta muy bien a los puntos, pero mucha varianza, ya que al resamplear se obtendría una curva muy distinta. En el centro se muestra el resultado entregado por un modelo con muy poca capacidad, que presentaría poca varianza pero mucho sesgo. A la derecha se muestra cómo se vería un resultado con poco sesgo y poca varianza. (Sanz, 2021)|


Para comprender lo que está ocurriendo de mejor manera, programemos el ejemplo mencionado en python, y veamos los resultados obtenidos con una capacidad baja, una apropiada y una alta. En este código, primero se define la función que usaremos para generar nuestros datos, que será la parábola **$y=f(x)=-(x-5)^2+5$**. Luego se definen las funciones que se usarán para entrenar el modelo a partir de un set de datos. Después, se generan 25 valores distribuidos uniformemente entre 0 y 10, los que pasan por la función **$f(x)$** y luego se les aplica ruido Gaussiano. Estos datos son usados para entrenar el modelo 3 veces, primero para ajustar un polinomio de grado 1 (poca capacidad), después de grado 2 (capacidad apropiada) y por último de grado 10 (mucha capacidad). Repitamos todo este experimento tres veces y grafiquemos los resultados obtenidos.

```python
# parte de este código está basado en la tarea 1

import random
import numpy as np
from matplotlib import pyplot as plt


_x_train = []                                                         # aquí guardaremos los valores generados
_y_train = []


def f(x):
  return -(x - 5)**2 + 5                                              # los datos se generan a partir de una parabola con ruido


def polyfit(x, y, degree):
  xT = x[:, np.newaxis]
  X = np.power(xT, np.linspace(0, degree, degree + 1))
  XT = np.matrix.transpose(X)
  w = np.matmul(np.matmul(np.linalg.inv(np.matmul(XT, X)), XT), y)
  return w

def poly1d(x, w):
  degree = len(w) - 1
  xT = x[:, np.newaxis]
  X = np.power(xT, np.linspace(0, degree, degree + 1))
  y = np.dot(X, w)
  return y


for i in range(25):                                                   # generamos los 25 puntos
  numero = random.uniform(0, 10)
  ruido = np.random.normal(0, 3)
  _x_train.append(numero)
  _y_train.append(f(numero) + ruido) # generamos el training set

x_train = np.array(_x_train)                                          # convertimos las listas en arrays
y_train = np.array(_y_train)

w1 = polyfit(x_train, y_train, degree=1)
w2 = polyfit(x_train, y_train, degree=2)
w10 = polyfit(x_train, y_train, degree=10)

x_fit = np.linspace(0.5,9.5,100)                                      # generamos las 3 curvas obtenidas con distintas capacidades
y_fit1 = poly1d(x_fit, w1)
y_fit2 = poly1d(x_fit, w2)
y_fit10 = poly1d(x_fit, w10)



plt.scatter(x_train, y_train, label="train set", s=10)
plt.plot(np.linspace(0,10,100), f(np.linspace(0,10,100)), label="f(x)", linewidth=0.5)
plt.plot(x_fit, y_fit1, label="degree=1", linewidth=0.5)
plt.plot(x_fit, y_fit2, label="degree=2", linewidth=0.5)
plt.plot(x_fit, y_fit10, label="degree=10", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()                                                          # graficamos los resultados

```

| ![a](https://github.com/jabarzuar/PIML/assets/101306821/b6da7283-c1ab-48b5-a905-96a7c133f26b) | ![b](https://github.com/jabarzuar/PIML/assets/101306821/b2e16e20-7d9a-47a6-ad38-2b1e1927a3ec) | ![c](https://github.com/jabarzuar/PIML/assets/101306821/c5f06f7f-ebe1-47fc-99f2-6531a115492d) |
| --- | --- | --- |

|Figura 2: Se muestran los resltados de repetir el experimento tres veces. Observamos que en cada iteración la curva roja, que corresponde a alta capacidad, se ajusta muy bien a los puntos, incluso pasando casi exactamente por algunos de ellos. Sin embargo, la curva obtenida en cada iteración es muy distinta a las demás, lo que implica que no es una buena representación de la función usada para generar los datos, y tendría un mal rendimiento al enfrentarse a datos distintos; estas curvas tienen poco sesgo (se alejan poco de los datos), pero mucha varianza (varían mucho entre sí). Por otro lado, vemos que la curva naranja, de baja capacidad, no se ajusta bien a los puntos, pero el resultado obtenido es similar en cada iteración; tienen mucho sesgo, pero poca varianza. Por último, la curva verde tiene la capacidad apropiada para obtener un buen equilibrio entre sesgo y varianza, vemos que son mas o menos similares en cada iteración y se ajustan decentemente a los valores. Además vemos que la curva generada con la capacidad adecuada es la que más se asemeja a la función real, mostrada en azul. |
|:------|



Es claro que, para un modelo que debería idealmente obtener una parábola, ninguno de los dos primeros resultados son algo deseable. El error presente en la curva de grado 10 se denomina _overfitting_, y modelos que lo presenten tendrán muy poco error de _training_, pero muy alto error de _testing_, basicamente el modelo está memorizando los datos de _training_ de manera que se ajusta muy bien a ellos, pero al cambiar estos datos el modelo ya no rinde bien. Por otro lado, la curva de grado 1 presenta un error llamado _underfitting_, lo que hace que el error de _training_ sea muy alto, ya que el modelo no logra ajustar bien a la función. Como vimos, tanto el sesgo como la varianza están relacionados con la capacidad del modelo, lo que sugiere que no es posible minimizar ambos a la vez; reducir la varianza implica aumentar el sesgo y viceversa. Esta disyuntiva se conoce como _**variance-bias tradeoff**_ o **compensación de varianza y sesgo**, lo que quiere decir que se debe encontrar un punto medio entre minimizar el sesgo y minimizar la varianza, o como lo define Huilgol, "el objetivo es encontrar el nivel correcto de complejidad en un modelo para minimiazr tanto el sesgo como la varianza, logrando buena generalización para datos nuevos" (2023). 

| ![Captura de pantalla 2023-10-15 031158](https://github.com/jabarzuar/PIML/assets/101306821/37920deb-ef02-4586-bc0b-25cd383c9a94)  
|:---|
|Figura 3: Se muestra la relación típica entre capacidad (eje horizontal) y error (eje vertical). Se observa que el error de _training_ disminuye de forma continua a medida que se aumenta la capacidad, a diferencia del error de generalización que alcanza un mínimo en cierto punto y luego aumenta debido al _overfitting_ (generalización es la capacidad de un modelo de rendir bien frente a nuevos inputs). La capacidad óptima del modelo se encuentra entre la zona de _underfitting_ y _overfitting_ (Goodfellow, 2016).

En _machine learning_ existe un teorema llamado **"_no free lunch theorem_"**, que afirma que "cualquier par de modelos es equivalente cuando su rendimiento es promediado en todos los posibles problemas" (Wolpert, 2013), lo que se puede entender como que no existe un algoritmo que funcione bien en todas las tareas. Esto implica que un modelo debe ser diseñado con una tarea específica en mente, para rendir bien en ésta. En el ejemplo anterior, se mencionó que una forma de medir el rendimiento de un modelo es según el **error cuadrático medio**, el que se calcula como **$MSE = E\[(\hat{\theta}_m - \theta)^2] = Bias(\hat{\theta}_m)^2 - Var(\hat{\theta}_m)$**; como podemos ver, minimizar el error cuadrático medio implica reducir tanto el sesgo como la varianza. Sin embargo, es posible que esto no sea suficiente para obtener la capacidad óptima, es por esto que existe la regularización. La **regularización** se refiere a toda modificación que se hace al modelo para reducir su error de generalización. Un ejemplo de esto es _weight decay_, o regularización L2, que consiste en agregar un término **$\lambda \omega^T \omega$**, llamado regularizador, en que **$\omega$** es el vector de pesos, a la función de costo, de manera que, dependiendo del valor que se le asigne a **$lambda$** se priorizarán pesos más pequeños, ayudando a prevenir _overfitting_; en este caso, valores mayores de **$\lambda$** resultarán en mayor regularización, mientras que valores cercanos a 0 tendrán poco efecto. Otros tipos de regularización usan distintos regularizadores, pero cumplen el fin de reducir el error de generalización buscando ajustar la capacidad para llegar al equilibrio entre error de sesgo y de varianza.

En conclusión, los modelos predictivos de _machine learning_ son propensos a error de **sesgo** y de **varianza**. El error de sesgo se refiere a la diferencia entre el el valor esperado de la estimación y el valor real, y ocurre por _underfitting_. El error de varianza se refiere al cambio exagerado de los resultados al cambiar el _input_, y ocurre por _overfitting_. Encontrar la capacidad adecuada para un modelo es fundamental si se quiere evitar caer en estos errores, lo que se demostró con un ejemplo práctico. Existen distintas formas de medir el rendimiento de un modelo, como lo es el error cuadrático medio, cuya fórmula incluye el sesgo y la varianza. El uso de regularizadores también es útil para llegar a una capacidad apropiada de nuestro modelo; el uso de estas herramientas es necesario debido a que, de acuerdo con el _no free lunch theorem_, el diseño de cada modelo dependerá de la función que tendrá que cumplir.


## Referencias


Goodfellow, I., Bengio, Y. & Courville, A. (2016). _Deep Learning_. MIT Press. https://www.deeplearningbook.org/

Huilgol, P. (14 de Agosto, 2023). _Bias and Variance in Machine Learning – A Fantastic Guide for Beginners!_. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2020/08/bias-and-variance-tradeoff-machine-learning/

Sanz, F. (2021). _Entiende de una vez qué es el Tradeoff Bias-Variance_. The Machine Learners. https://www.themachinelearners.com/tradeoff-bias-variance/#Encontrando_el_balance_perfecto

Wolpert, D. (23 de Agosto, 2013). _Coevolutionary Free Lunches_. NASA Technical Reports Center. https://ntrs.nasa.gov/citations/20060007558

