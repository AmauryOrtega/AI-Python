import pandas
import matplotlib.pyplot as graficar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as ACP
from sklearn.cross_validation import train_test_split as separar
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as ECM
from sklearn.ensemble import RandomForestRegressor

GRAFICAR = True

juegos = pandas.read_csv("games.csv")


print("-"*5 + "Columnas" + "-"*5)
print(juegos.columns)
print()
print("-"*5 + "Tamaño de DataSet (Filas, Columnas)" + "-"*5)
print(juegos.shape)
print()

if GRAFICAR:
    graficar.title("Distribucion de puntuacion promedio")
    graficar.xlabel("Puntuacion promedio")
    graficar.ylabel("# de Juegos")
    graficar.hist(juegos["average_rating"])
    graficar.show()
    graficar.clf()

print("-"*5 + "Diff entre juego con puntaje de 0 y con puntaje superior a 0" + "-"*5)
print(juegos[juegos["average_rating"] == 0].iloc[0])
print(juegos[juegos["average_rating"] > 0].iloc[0])
print()

juegos = juegos[juegos["users_rated"] > 0]
juegos = juegos.dropna(axis=0)

if GRAFICAR:
    graficar.title("Distribucion de puntuacion promedio")
    graficar.xlabel("Puntuacion promedio")
    graficar.ylabel("# de Juegos")
    graficar.hist(juegos["average_rating"])
    graficar.show()
    graficar.clf()

print("-"*5 + "Tamaño de DataSet (Filas, Columnas)" + "-"*5)
print(juegos.shape)
print()

modelo_kmeans = KMeans(n_clusters=5, random_state=1)
columnas_numero = juegos._get_numeric_data()
modelo_kmeans.fit(columnas_numero)
etiquetas = modelo_kmeans.labels_
acp_2 = ACP(2)
columnas_a_graficar = acp_2.fit_transform(columnas_numero)

if GRAFICAR:
    graficar.title("Agrupacion de juegos en 5 clusters con ACP")
    graficar.scatter(x=columnas_a_graficar[:,0], y=columnas_a_graficar[:,1], c=etiquetas)
    graficar.show()
    graficar.clf()

print("-"*5 + "Correlacion de average_rating" + "-"*5)
print(juegos.corr()["average_rating"])
print()

columnas = juegos.columns.tolist()
columnas = [columna for columna in columnas if columna not in ["bayes_average_rating", "average_rating", "type", "name"]]
columna_a_predecir = "average_rating"
set_entrenamiento = juegos.sample(frac=0.8, random_state=1)
set_test = juegos.loc[~juegos.index.isin(set_entrenamiento.index)]

print("-"*5 + "Tamaño de set_entrenamiento (Filas, Columnas)" + "-"*5)
print(set_entrenamiento.shape)
print()
print("-"*5 + "Tamaño de set_test (Filas, Columnas)" + "-"*5)
print(set_test.shape)
print()

modelo = LinearRegression()
modelo.fit(set_entrenamiento[columnas], set_entrenamiento[columna_a_predecir])
predicciones = modelo.predict(set_test[columnas])

print("-"*5 + "Predicciones" + "-"*5)
print(predicciones)
print("-"*5 + "VS" + "-"*5)
print(juegos.tail(1)["average_rating"])
print()

print("-"*5 + "Error en prediccion" + "-"*5)
print(ECM(predicciones, set_test[columna_a_predecir]))
print()

if GRAFICAR:
    graficar.figure("lineal")
    graficar.title("Regresion lineal")
    graficar.xlabel("ID Juego")
    graficar.ylabel("Puntuacion promedio")
    graficar.scatter(set_test["id"], set_test["average_rating"], label="Real")
    graficar.scatter(set_test["id"], predicciones, label="Prediccion")
    graficar.legend(loc="upper left")

print("-"*5 + "Usando RANDOM FOREST" + "-"*5)

modelo = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
modelo.fit(set_entrenamiento[columnas], set_entrenamiento[columna_a_predecir])
predicciones = modelo.predict(set_test[columnas])

print("-"*5 + "Predicciones" + "-"*5)
print(predicciones)
print("-"*5 + "VS" + "-"*5)
print(juegos.tail(1)["average_rating"])
print()
print("-"*5 + "Error en prediccion" + "-"*5)
print(ECM(predicciones, set_test[columna_a_predecir]))
print()

if GRAFICAR:
    graficar.figure("random")
    graficar.title("Regresion Random Forest")
    graficar.xlabel("ID Juego")
    graficar.ylabel("Puntuacion promedio")
    graficar.scatter(set_test["id"], set_test["average_rating"], label="Real")
    graficar.scatter(set_test["id"], predicciones, label="Prediccion")
    graficar.legend(loc="upper left")
    graficar.show()
