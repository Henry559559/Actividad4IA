# Importación de bibliotecas necesarias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Carga de datos
# Asegúrese de tener un archivo CSV con datos relevantes, o reemplace esta sección con la carga de datos adecuada.
ruta_del_archivo = 'datos.csv'
df = pd.read_csv(ruta_del_archivo)

# Preprocesamiento de datos (si es necesario)
# Realice cualquier limpieza y transformación necesaria en los datos
# Por ejemplo, eliminar valores nulos, codificar variables categóricas, etc.

# Selección del método de agrupamiento y ajuste del modelo
# En este caso, estamos usando K-means
n_clusters = 3  # Ajuste el número de clusters según sea necesario
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(df)

# Asignación de etiquetas de cluster a los datos
df['Cluster'] = kmeans.labels_

# Evaluación del modelo
coeficiente_silueta = silhouette_score(df, kmeans.labels_)
print(f'Coeficiente de Silueta: {coeficiente_silueta}')

# Visualización de los clusters
# Asumiendo que estamos trabajando con un dataset 2D para simplificar la visualización
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis', label='Datos')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroides')
plt.title('Clusters y Centroides')
plt.legend()
plt.show()

# Guardar el DataFrame con las etiquetas de cluster a un nuevo archivo CSV
df.to_csv('datos_con_clusters.csv', index=False)
