## Instalar
```
sudo pip install pandas
sudo pip install matplotlib
sudo pip install scipy scikit-learn
```

## DataSet
Proveniente de [BoardGamesGeek](http://www.boardgamegeek.com/), [Sean Beck](https://github.com/ThaWeatherman) convirtio la siguiente informacion de 8000 tableros en formato csv.

* `name` - Nombre del juego.
* `playingtime` -  Tiempo de juego (Por el creador del juego).
* `minplaytime` - Tiempo minimo de juego (Por el creador del juego).
* `maxplaytime` - Tiempo maximo de juego (Por el creador del juego).
* `minage` - Edad minima recomendada para jugar.
* `users_rated` - Numero de usuarios que calificaron el juego.
* `average_rating` - Calificacion promedio de los usuarios (1-10).
* `total_weights` - Numero de Weights dados por los usuarios. Es una medida de que tan complicado es el juego.
* `average_weight` - Weight promedio (0-5).

## Pandas
Leer e imprimir resumenes estadisticos de un DataSet. La libreria ofrece estructuras de datos y herramientas de analisis de datos. Se usara la estructura de datos *dataframe*.
Actualmente la informacion se ve asi.
```
id,type,name,yearpublished,minplayers,maxplayers,playingtime
12333,boardgame,Twilight Struggle,2005,2,2,180
120677,boardgame,Terra Mystica,2012,2,5,150
```
|    |1       |2           |3                   |4              |
|--- |--------| -       ---|                ----|----          -|
|1   |id      |type        |name                |yearpublished  |
|2   |12333   |boardgame   |Twilight Struggle   |2005           |
|3   |120677  |boardgame   |Terra Mystica       |2012           |

Las matrices se pueden tratar usando NumPy pero tienes desventajas. Asi que se usa *dataframe*.
