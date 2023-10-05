# Importar las bibliotecas necesarias
from estadoff import candidatos_df
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import pandas as pd


# Cargar los datos desde el archivo Excel
#data = pd.read_excel("1000-Registros-de-ventas.xlsx")
print('1')
data= candidatos_df
data=data[[ "acousticness","danceability " ,"duration_ms"]]
print(data.head())  # Verificar los datos cargados


# Dividir el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

# Crear un modelo Nearest Neighbors (vecinos más cercanos)
model = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')

# Entrenar el modelo en los datos de entrenamiento
model.fit(X_train)

# Realizar recomendaciones para un usuario específico (por ejemplo, el usuario 1)
user_to_recommend = X_test.iloc[0].values.reshape(1, -1)
distances, indices = model.kneighbors(user_to_recommend, n_neighbors=2)

# Imprimir las recomendaciones
print("Recomendaciones para el usuario 1:")
for i in indices[0]:
    if data.iloc[i].sum() == 0:  # Evitar recomendar canciones ya escuchadas
        print(f"Canción {i + 1}")
