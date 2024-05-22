# Objetivos
#- Aplicar con éxito todos los conocimientos que has adquirido a lo largo del Bootcamp.
#- Consolidar las técnicas de limpieza, entrenamiento, graficación y ajuste a modelos de *Machine Learning*.
#- Generar una API que brinde predicciones como resultado a partir de datos enviados.


# Proyecto

#1. Selecciona uno de los siguientes *datasets*:
  #- *Reviews* de aplicaciones de la Google Play Store: https://www.kaggle.com/datasets/lava18/google-play-store-apps
  #- Estadísticas demográficas de los ganadores del premio Oscar de la Academia: https://www.kaggle.com/datasets/fmejia21/demographics-of-academy-awards-oscars-winners
 # - Aspiraciones profesionales de la generación Z: https://www.kaggle.com/datasets/kulturehire/understanding-career-aspirations-of-genz

#Cada uno representa un *dataset*, un problema y una forma diferente de abordarlo. Tu tarea es identificar las técnicas y modelos que podrías usar para tu proyecto.

#2. Debes hacer un análisis exploratorio y limpieza de los datos. Usa las ténicas que creas convenientes.

#3. Entrena el modelo de *Machine Learning*, procesamiento de lenguaje natural o red neuronal que creas adecuado.

#4. Genera por lo menos dos gráficas y dos métricas de rendimiento; explica las puntuaciones de rendimiento que amerite tu problema. Todas las gráficas de rendimiento que realices deben tener leyendas, colores y títulos personalizados por ti.

 # - Además, antes de subir el modelo a "producción", deberás realizar un proceso de ensambles (*ensemblings*) y de ajuste de hiperparámetros o *tuning* para intentar mejorar la precisión y disminuir la varianza de tu modelo.

#5. Construye una API REST en la que cualquier usuario pueda mandar datos y que esta misma devuelva la predicción del modelo que has hecho. La API debe estar en la nube, ya sea en un servicio como Netlify o Ngrok, para que pueda ser consultada desde internet.

#6. Genera una presentación del problema y del modelo de solución que planteas. Muestra gráficas, datos de rendimiento y explicaciones. Esta presentación debe estar enfocada a personas que no sepan mucho de ciencia de datos e inteligencia artificial.

#7. **Solamente se recibirán trabajos subidos a tu cuenta de GitHub con un README.md apropiado que explique tu proyecto**.

## Criterios de evaluación

#| Actividad | Porcentaje | Observaciones | Punto parcial
#| -- | -- | -- | -- |
#| Actividad 1. Limpieza y EDA | 20 | Realiza todas las tareas necesarias para hacer el EDA y la limpieza correcta, dependiendo de la problemática. Debes hacer como mínimo el análisis de completitud, escalamiento (si aplica) y tokenización (si aplica). | Realizaste solo algunas tareas de exploración y limpieza y el modelo se muestra aún con oportunidad de completitud, escalamiento y/o mejora. |
#| Actividad 2. Entrenamiento del modelo | 20 | Elige el modelo y algoritmo adecuados para tu problema, entrénalo con los datos ya limpios y genera algunas predicciones de prueba. | No has realizado predicciones de prueba para tu modelo de ML y/o tu modelo muestra una precisión menor al 60 %. |
#| Actividad 3. Graficación y métricas | 20 | Genera por lo menos dos gráficas y dos muestras de métricas que permitan visualizar el rendimiento y precisión del modelo que construiste. Además, realizaste los procesos de *tuning* y ensambles adecuados para tu problema. | Las gráficas no tienen leyendas y colores customizados, solo muestras una gráfica o no realizaste el *tuning* de hiperparámetros.
#| Actividad 4. API REST | 20 | Generaste con éxito un *link* público en el que, por método POST, se puede mandar información y la API REST devuelve una predicción junto con el porcentaje de confianza de esta misma. | N/A
#| Actividad 5. Presentación | 20 | Genera una presentación en la que establezcas como mínimo: el problema, proceso de solución, metodologías usadas, gráficas de rendimiento, demostración del modelo y aprendizajes obtenidos. Debes redactarla con términos que pueda entender cualquier persona, no solo científicos de datos. | La presentación no expone con claridad o en términos coloquiales el proceso de creación del modelo, sus ventajas y muestras de rendimiento.

##**Inicio**

#Se selecciona el siguiente data frame:
#1. Reviews de aplicaciones de la Google Play Store: https://www.kaggle.com/datasets/lava18/google-play-store-apps

###**Preguntas a responder**

#1. Puedo manejar el conjunto de datos con un solo modelo?
#2. Cual es el proposito de crear un modelo de ML, NPL, redes neuronales a un reviews de aplicaciones?
#3. Que datos son los mas importantes para cada modelo?
#4. Como elijo las variables o columnas para cada modelo?
#5. Puedo hacer un analisis exploratorio que me ayude a determinar las columnas que usare en cada modelo?

#Iniciamos importando las librerias iniciales para empezar la exploracion del data set
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

###**Analisis exploratorio de datos EDA**

#Se hace un analisis inicial de todo el data frame original para decidir la limpieza y preprocesado que se realizara.

# Se crea un data frame que contendra el archivo csv con la informacion a explorar
data_full = pd.read_csv('C:/Users/Luis Carlos Villalaz/OneDrive/Escritorio/U Camp/Data science and AI/Modulo 7/googleplaystore.csv')

# Los datos iniciales mostrados continen información sobre aplicaciones
# de Google Play Store, incluyendo el nombre de la aplicación ( App),
# categoría ( Category), calificación ( Rating), cantidad de compra (
# Reviews), tamaño ( Size), cantidad de instalaciones ( Installs), entre otros.
data_full.head(5)

#El resumen de la información nos indica que el conjunto de datos contiene 10,841 filas y 13 columnas.
#La columna Rating tiene valores nulos, lo que podría requerir manejo durante la limpieza de datos.
# Las columnas Reviews, Size, Installs, Price, Last Updated, Current Ver, y
# Android Ver están representadas como objetos (cadenas de texto) en lugar de
# numéricos, lo que puede necesitar conversión para su análisis.
data_full.info()

#La descripción estadística muestra algunos aspectos interesantes:
#La calificación ( Rating) tiene un promedio de aproximadamente 4,19 lo que indica que la calificacion en la app es alta (generalente siempres es de 5.0)
#con una desviación estándar de 0,54, lo que indica una dispersion bastante baja muy cercana al promedio.
#La columna App tiene valores únicos (9660 únicos de 10841 registros), lo que sugiere posibles duplicados.
#La columna Installs es un objeto y muestra la categoría de aplicaciones.
data_full.describe(include='all')

###**Conclusiones del analisis inicial de datos**

#Este data frame es muy diverso y no se puede trabajar completo con un solo modelo.

#Puedo destacar lo que podemos hacer en base a los datos revisados:

#1. Predicción de Calificaciones: El objetivo principal podría ser predecir la calificación (rating) de una aplicación en función de las características de su review. Esto podría ser útil para empresas que desean comprender la satisfacción del usuario y mejorar sus productos.

#2. Análisis de Sentimientos: El modelo podría utilizarse para realizar análisis de sentimientos sobre las opiniones de los usuarios hacia una aplicación. Esto podría revelar información valiosa sobre las preferencias y opiniones de los usuarios.

#3. Mejora de Productos y Servicios: Las empresas pueden utilizar estos modelos para identificar áreas de mejora en sus aplicaciones según el feedback de los usuarios, lo que podría ayudar a priorizar las actualizaciones y optimizaciones.

#4. Clasificación de Comentarios: El modelo podría clasificar automáticamente los comentarios en categorías como positivos, neutros o negativos, lo que permitiría un análisis más eficiente de grandes volúmenes de feedback.

##**1. Modelo de Prediccion de calificaciones**

###**Creacion de nuevo data frame**

#Se generara un nuevo Data Frame con las variables necesarias para la construccion del modelo predictivo.

#Queremos predecir la calificación de una aplicación en función de las características que seleccionaremos a continuacion. El objetivo es utilizar estas características para construir un modelo que pueda predecir la calificación de una aplicación en la Google Play Store.

#Con la creacion de este nuevo df nos aseguramos de poder manipular con confianza
#las variables, sin afectar las variables originales para el uso en otro modelo
# Se eligen las variables que se consideraron con mayor importancia para el modelo predictivo
calification = pd.DataFrame(data_full, columns=["Rating", "Reviews", "Installs", "Price"])

###**Analis exploratorio de datos EDA**
#Para el modelo predictivo.

#Revisamos las primeras columnas para ver su contenido
calification.head(5)

#Revisamos la informacion del nuevo data frame en donde podemos ver columnas con valores nulos
# Tambien podemos ver que el tipo de columnas que se tienen estan identificadas como object lo
# que nos puede dar problemas para su manejo
calification.info()

#Revisamos la informacion descriptiva
calification.describe(include='all')

# Contar valores nulos por columna
#Lo que nos muestra 1 variables con valores nulos y en los que debemos trabajar para la limpieza de los datos
null_counts = calification.isnull().sum()
print("Valores nulos por columna:")
print(null_counts)
print("\n")

# Mostraremos valores únicos por columna que nos ayudaran a revisar, limpiar y preprocesar nuestras variables.
for column in calification.columns:
    unique_values = calification[column].unique()
    print(f"Valores únicos en la columna '{column}':")
    print(unique_values)
    print("\n")


#Revisemos a detalle lo encontrado en nuestras variables:

#1. Valores únicos en la columna 'Rating':
#Esta es nuestra variable objetivo, que representa la calificación de la aplicación. Está en formato numérico y contiene valores desde 1.0 hasta 5.0, con algunos valores atípicos como 19.0.

#2. Valores únicos en la columna 'Reviews':
#Esta columna contiene el número de reseñas de la aplicación. Aunque es numérica, actualmente está almacenada como tipo de objeto (texto). Necesitaremos convertirla a un tipo numérico para utilizarla en el modelo.

#3. Valores únicos en la columna 'Installs':
#'Installs' indica el número de instalaciones de la aplicación. Los valores están en formato de texto con sufijo '+'. Necesitaremos limpiar estos datos y convertirlos a un formato numérico.

#4. Valores únicos en la columna 'Price':
#'Price' representa el precio de la aplicación. Los valores incluyen precios en dólares con el símbolo '$' y pueden contener decimales. Necesitaremos limpiar y convertir esta columna a un formato numérico.


###**Limpieza de datos**

#Como la variable 'Rating' sera nuestra variable objetivo, manejaremos bien nuestros valores nulos
# Calcular la media de Rating
mean_rating = calification['Rating'].mean()

# Imputar los valores nulos en Rating con la media
calification['Rating'].fillna(mean_rating, inplace=True)

# Verificar valores nulos después de la imputación y eliminación
null_counts_updated = calification.isnull().sum()
print("Valores nulos por columna después de manejarlos:")
print(null_counts_updated)


# Convertimos 'Reviews' a tipo numérico removiendo los simbolos y la nomesclatura de kilo y mega a numeros para que podamos usarla en el modelo
calification['Reviews'] = calification['Reviews'].replace('[KM]+$', '', regex=True).astype(float) * calification['Reviews'].str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int)

# Limpiamos 'Installs' para convertirlo a tipo numérico
# Reemplazar valores no numéricos con NaN
calification['Installs'] = calification['Installs'].replace('Free', np.nan)

# Eliminar caracteres no numéricos
calification['Installs'] = calification['Installs'].str.replace('[^0-9]', '', regex=True)

# Convertir a tipo float
calification['Installs'] = calification['Installs'].astype(float)

# Removemos el símbolo '$' y convertimos 'Price' a valores numéricos
# Convertir 'Price' a tipo numérico
def convert_price(price):
    if price == 'Everyone':
        return 0
    else:
        return float(price.strip('$'))

calification['Price'] = calification['Price'].apply(convert_price)

# Revisamos la informacion de las variables en nuestro data frame
calification.info()

for column in calification.columns:
    unique_values = calification[column].unique()
    print(f"Valores únicos en la columna '{column}':")
    print(unique_values)
    print("\n")

#Como se ven en los 2 bloques de codigo anteriores, en la variable 'Installs' tenemos 1 NaN que debemos tratar

#Eliminamos filas con NaN en la columna 'Installs'
calification.dropna(subset=['Installs'], inplace=True)

#Calculamos la media de la columna 'Installs'
installs_mean = calification['Installs'].mean()

# Rellenamos los NaN en la columna 'Installs' con la media calculada
calification['Installs'].fillna(installs_mean, inplace=True)

##**Visualizaciones**

# Ajusta el tamaño y estilo del gráfico
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')  # Establece un estilo de gráfico, por ejemplo, 'ggplot'

# Crea el gráfico de caja con las variables seleccionadas
fig, ax = plt.subplots()
ax.boxplot(calification)

# Personalizacion de las etiquetas y el título del gráfico
plt.title('Distribución de Variables Numéricas con Boxplot')
plt.xticks(rotation=45)
column_names = list(calification.columns)
ax.set_xticklabels(column_names, rotation=45)
plt.xlabel('Variables')
plt.ylabel('Valores')

# Muestra el gráfico
plt.show()

#Aunque el grafico de caja y bigotes muestra valores atipicos en 2 de sus variables, no podemos decir que sean ouliers ya que ambas variables manejan altos valores.

#Se hicieron purbas con ajustando los valores atipicos de todas las variables asi como tambien de solo 2 y aun asi el modelo no mejoro.

# Matriz de correlación
corr = calification.corr()

# Ajustamos el tamaño de la figura
plt.figure(figsize=(10, 8))

# Creamos un mapa de calor con la matriz de correlación
heatmap = sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)

# Personalizamos las etiquetas y el título del gráfico
plt.title('Matriz de Correlación')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Muestramos el gráfico
plt.show()

#En este mapa de calor se nota una mejor correlacion entre la variable installs y reviews.

sns.pairplot(calification[['Rating', 'Reviews', 'Installs', 'Price']])
plt.suptitle('Matriz de Dispersión')
plt.show()


#Con esta matriz de dispersion podemos comparar diferentes variables y su correlacion.

###**Modelo base Random Forest Regressor y ensamble GradientBoostingRegressor**

# Importamos las librerias necesarias para nuestro modelo predictivo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Definimos nuestros datos en variable (X), variable (y)
X = calification.drop('Rating', axis=1)
y = calification['Rating']

# Dividimos nuestros datos en un conjunto de entrenamiento y uno de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definimos los modelos base y de ensamblado
base_model = RandomForestRegressor(random_state=42)
ensemble_model = GradientBoostingRegressor(random_state=42)

###**Tunning de modelo base**

# Definimos los hiperparámetros a ajustar para el modelo base
base_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [50, 100, 200],
    'min_samples_split': [2, 5, 10, 20, 50]
}

# Realizamos la búsqueda de cuadrícula para el modelo base
base_grid = GridSearchCV(base_model, param_grid=base_params, scoring='neg_mean_squared_error', cv=5)
base_grid.fit(X_train, y_train)

###**Tunning de modelo de ensamble**

# Definimos los hiperparámetros a ajustar para el modelo ensamble
ensemble_params = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.1, 0.5, 1.0]
}


# Realizamos la búsqueda de cuadrícula para el modelo de ensamblado
ensemble_grid = GridSearchCV(ensemble_model, param_grid=ensemble_params, scoring='neg_mean_squared_error', cv=5)
ensemble_grid.fit(X_train, y_train)

###**Entrenamiento de modelos**

# Obtenemos los mejores modelos
best_base_model = base_grid.best_estimator_
best_ensemble_model = ensemble_grid.best_estimator_

# Entrenamos los mejores modelos
best_base_model.fit(X_train, y_train)
best_ensemble_model.fit(X_train, y_train)

# Hacemos las predicciones con ambos modelos
y_pred_base = best_base_model.predict(X_test)
y_pred_ensemble = best_ensemble_model.predict(X_test)

###**Metricas de evaluacion**

# Evaluamos el modelo ensamblado
mse_base = mean_squared_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

print("Base Model - Mean Squared Error:", mse_base)
print("Base Model - R2 Score:", r2_base)

print("Ensemble Model - Mean Squared Error:", mse_ensemble)
print("Ensemble Model - R2 Score:", r2_ensemble)

###**Visualizacion**

# Gráfico de dispersión de predicciones vs. valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_ensemble, alpha=0.5)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs. Valores reales")
plt.show()

# Gráfico de línea para visualizar cómo se comportan las predicciones vs. los valores reales
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(y_test)), y_test, label='Valores reales', marker='o')
plt.plot(np.arange(len(y_test)), y_pred_ensemble, label='Predicciones', marker='x')
plt.xlabel("Índice de muestra")
plt.ylabel("Valor")
plt.title("Comparación de Predicciones y Valores reales")
plt.legend()
plt.show()

###**Conclusiones:**

#Despues de realizar un analisis, limpieza y preprocesado en el conjunto de datos; Realizamos el entrenamiento de nuestro de modelo de regresion con RandomForestRegressor, sin embargo la eficiencia no fue buena.

#Lo intentamos con otros modelos tales como

#GradientBoostingRegressor

#ExtraTreesRegressor

#AdaBoostRegressor

#LinearRegression

#BaggingRegressor

#StackingRegressor

#DecisionTreeRegressor

#KNeighborsRegressor

#Support Vector Machines (SVM)

#Se uso GridSearchCV para una busqueda del tunning mas eficiente


#Sin embargo ninguno paso de 12% de eficiencia.
#Algunos empeoraron mas que otros.


#Todas esta pruebas me indican que este modelo no es capaz de hacer este tipo de predicciones y se debe intentar con otro modelo ya sea redes neuronales, deep learning o NLP.

##**Entorno local**

#Para trabajar en un entorno local y generar nuestro archivo app.py y poder generar flask app se debe transformar el modelo en un archivo .pkl

#import joblib
# Guardar el modelo entrenado utilizando joblib
#joblib.dump(ensemble_model, 'modelo_calificacion.pkl')