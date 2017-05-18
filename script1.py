# Import librerias
import pandas
import matplotlib.pyplot as graficar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as ACP
from sklearn.cross_validation import train_test_split as separar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as ECM
from sklearn.ensemble import RandomForestRegressor

# Varibale global
GRAFICAR = False

juegos = pandas.read_csv("games.csv")
# Imprime las columnas que se leyeron
print("-"*5 + "Columnas" + "-"*5)
print(juegos.columns)
print()
# Imprime cuantas filas, columnas tenemos
print("-"*5 + "Tamaño de DataSet (Filas, Columnas)" + "-"*5)
print(juegos.shape)
print()

# Suponiendo que queremos predecir que puntuacion en promedio
# le harian los usuarios a un juego que aun no ha salido. Esta informacion
# se encuentra en la columna average_rating

# Hacemos un histograma de esta columna para saber la distribucion
# de las puntiaciones en promedio de todos los juegos usando
# Lee DataSet
# indexacion por columna que retorna toda una columna
graficar.hist(juegos["average_rating"])
if GRAFICAR: graficar.show()

# juegos[juegos["average_rating"] == 0] retornara un dataframe con solo las
# filas donde el valor de la columna average_rating es 0

# Indexando por posicion, podemos obtener toda una fila
# juegos.iloc[0] retornara toda la primera fila del dataframe juegos
# juegos.iloc[0,0] retornara la primera columna de la primera fila del dataframe
print("-"*5 + "Diff entre juego con puntaje de 0 y con puntaje superior a 0" + "-"*5)
print(juegos[juegos["average_rating"] == 0].iloc[0])
print(juegos[juegos["average_rating"] > 0].iloc[0])
print()
# Se determina que deben haber muchos juegos con 0 puntuaciones de usuarios
# por lo tanto el promedio de puntuaciones es 0

# Esta informacion se considera basura asi que se opta por eliminar los juegos
# que no hayan sido puntuados por algun usuarios
juegos = juegos[juegos["users_rated"] > 0]
# Remueve cualquier fila que le hagan falta valores
juegos = juegos.dropna(axis=0)

# Distribucion de puntuacion promedio
graficar.hist(juegos["average_rating"])
if GRAFICAR: graficar.show()
# Imprime cuantas filas, columnas tenemos
print("-"*5 + "Tamaño de DataSet (Filas, Columnas)" + "-"*5)
print(juegos.shape)
print()


# Análisis de grupos o agrupamiento
# es la tarea de agrupar un conjunto de objetos de tal manera que los miembros
# del mismo grupo (llamado clúster) sean más similares, en algún sentido u otro.
#EXPO
#Es la tarea principal de la minería de datos exploratoria y es una técnica común en el análisis de datos estadísticos. Además es utilizada en múltiples campos como el aprendizaje automático, el reconocimiento de patrones, el análisis de imágenes, la búsqueda y recuperación de información, la bioinformática, la compresión de datos y la computación gráfica.

# Un ejemplo de grupo son los juegos que no tenian puntuacion

# Se usara K-means, un metodo de agrupamiento donde cada elemento hará parte
# cuyo valor promedio se acerque mas
#EXPO
# K-means es un método de agrupamiento, que tiene como objetivo la partición de un conjunto de n observaciones en k grupos en el que cada observación pertenece al grupo cuyo valor medio es más cercano. Es un método utilizado en minería de datos.

# Se crea el modelo con 5 clusters y una semilla random de 1
modelo_kmeans = KMeans(n_clusters=5, random_state=1)
# Se quitan todos los tipos de datos que no sean numericos
columnas_numero = juegos._get_numeric_data()
# Se agrega la informacion al modelo
modelo_kmeans.fit(columnas_numero)
# Se obtienen las etiquetas de los clusters
etiquetas = modelo_kmeans.labels_

# Para visualizar los clusters o grupos, es necesario reducir el numero de
# columnas debido a que cada columna aumentara el grafico en 1 dimension
# asi que se usa Análisis de Componentes Principales (En español ACP, en inglés, PCA)
# es una tecnica para reducir la dimensionalidad de un conjunto de datos usando
# correlacion entre columnas

# Se crea modelo ACP
acp_2 = ACP(2)
# Se obtienen que columnas graficar
columnas_a_graficar = acp_2.fit_transform(columnas_numero)
# Se crea la grafica
graficar.scatter(x=columnas_a_graficar[:,0], y=columnas_a_graficar[:,1], c=etiquetas)
if GRAFICAR: graficar.show()

# Inteligencia artificial
# Para esto hay que determinar como se medira el error y que se va a predecir

# PREDECIR -> average_rating o el puntaje promedio de un juego

# ERROR
# Aqui se tiene en cuenta que se esta haciendo
# Regresion & variables continuas != Clasificacion & variables discretas

# En este caso se usara Error Cuadrático Medio (En español ECM, en inglés, MSE)
# porque es rapido de calcular y determina el promedio de que tan distantes
# estan las predicciones de los valores reales


# CORRELACION
# Sabiendo que se quiere predecir average_rating o el puntaje promedio de un juego
# es momento de decidir que columnas son de mayor interes para esto.
# Para esto se calculara la correlacion entre average_rating y el resto de columnas
print("-"*5 + "Correlacion de average_rating" + "-"*5)
print(juegos.corr()["average_rating"])
print()
# De aqui podemos decir que id y average_weight tienen mayor correlacion
# [ID]
# Suponiendo que este valor es dado cuando se agrega un juego
# es posible que los juegos mas nuevos tienen mejores puntuaciones
# tal vez al principio de BoardGameGeek los usuarios eran menos amables
# o que los juegos viejos tenian menos calidad
# [average_weight]
# Es posible que los juegos mas complejos hayan sido puntuados mejor

# Columnas para predecir
# Hay que remover las columnas no numericas
# Hay que remover las columnas que se calculen usando la columna a predecir average_rating
# se quitan "bayes_average_rating", "average_rating", "type", "name"

# Obtiene lista de columnas
columnas = juegos.columns.tolist()
# Filtrado de columnas, lo cual nos da los predictores
columnas = [columna for columna in columnas if columna not in ["bayes_average_rating", "average_rating", "type", "name"]]
# Se guarda la columna que se intentara predecir
columna_a_predecir = "average_rating"

# Es necesario separar el DataSet que se tiene en set para entrenamiento y set para pruebas
# Si no se hace, se consigue overfitting o sobre-ajuste que es sobre-entrenar un algoritmo
# de aprendizaje con un cierto set para los cuales ya conoce el resultado
# Ej: Si aprendes 1+1=2 y 2+2=4, seras capaz de responder con 0 errores
# Pero si te preguntan por 3+3, no seras capaz de resolverlo
# Por eso es necesario aprender de forma general

# Como norma, si el algoritmo de aprendizaje produce una cantidad de errores baja
# es recomendable revisar que no se este presentando un sobre-ajuste

# En este caso se usara el 80% del DataSet para entrenar y el 20% para probar
# Se crea el set de entrenamiento y de pruebas
set_entrenamiento = juegos.sample(frac=0.8, random_state=1)
set_test = juegos.loc[~juegos.index.isin(set_entrenamiento.index)]
# Imprime tamaño de ambos sets
print("-"*5 + "Tamaño de set_entrenamiento (Filas, Columnas)" + "-"*5)
print(set_entrenamiento.shape)
print()
print("-"*5 + "Tamaño de set_test (Filas, Columnas)" + "-"*5)
print(set_test.shape)
print()

# Se crea el modelo
modelo = LinearRegression()
# Se añaden los DataSets al modelo, el primero son los predictores y el segundo, el objetivo
modelo.fit(set_entrenamiento[columnas], set_entrenamiento[columna_a_predecir])
# Se crean predicciones
predicciones = modelo.predict(set_test[columnas])
print("-"*5 + "Predicciones" + "-"*5)
print(predicciones)
print("-"*5 + "VS" + "-"*5)
print(juegos.tail(1)["average_rating"])
print()
# Calcula error entre prediccion y los valores reales
print("-"*5 + "Error en prediccion" + "-"*5)
print(ECM(predicciones, set_test[columna_a_predecir]))
print()

# FIN DE REGRESSION LINEAL
# Aunque Scikit-learn nos permite usar otro algoritmo, se usara
# random forest que es capaz de encontrar correlaciones entre DataSets no lineales
# cosa que la Regresion lineal no seria capaz
# EJ: si minage o edad minima para un juego afecta a la puntuacion
# edad < 5, el puntaje es bajo
# edad 5-10, el puntaje es alto
# edad 10-15, el puntaje es bajo
print("-"*5 + "Usando RANDOM FOREST" + "-"*5)
# Se crea el modelo
modelo = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Se pasan los DataSets
modelo.fit(set_entrenamiento[columnas], set_entrenamiento[columna_a_predecir])
# Se hace la prediccion
predicciones = modelo.predict(set_test[columnas])
print("-"*5 + "Predicciones" + "-"*5)
print(predicciones)
print("-"*5 + "VS" + "-"*5)
print(juegos.tail(1)["average_rating"])
print()
# Calcula el error
print("-"*5 + "Error en prediccion" + "-"*5)
print(ECM(predicciones, set_test[columna_a_predecir]))
print()
