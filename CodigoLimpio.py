# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:11:00 2023

@author: Xavico
"""

#!pip install pyclustering
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools

# Cargar datos desde un archivo CSV
data = pd.read_csv('data/tabulacion de_imagenes_en_base_a_colores_multiespectrales.csv')
# Respaldar la segunda columna
columna_respaldo = data.iloc[:, 1]
# Eliminar las primeras seis columnas
data = data.iloc[:, 7:]
# Agregar nombres de columna personalizados
column_names = ['Columna{}'.format(i) for i in range(1, 521)]
data.columns = column_names
# Manejar valores faltantes si es necesario
data = data.fillna(0)  # Por ejemplo, rellenar valores faltantes con ceros

# Aplicar PCA para reducir la dimensionalidad
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
# Definir el numero de Cluster
num_clusters = 4
#colores de base
colors = ['g', 'y', 'b', 'r'] 


# Graficar la clasificacion originial,
for i in range(len(data_pca)):
    plt.scatter(data_pca[i, 0], data_pca[i, 1], color=colors[int(columna_respaldo[i]) % len(colors)], alpha=0.5)
plt.title('Clasificacion original')
plt.savefig('C:/PythonResultado/original.png')



# Aplicar K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data_pca)
cluster_labels_Kmeans = kmeans.labels_
# Visualizar los resultados

for i in range(len(data_pca)):
    plt.scatter(data_pca[i, 0], data_pca[i, 1], color=colors[int(cluster_labels_Kmeans[i]) % len(colors)], alpha=0.5)
plt.title('PCA + K-means Clustering')
# Calcular precisión
precision = accuracy_score(columna_respaldo, cluster_labels_Kmeans)
plt.text(0.05, 0.95, f"Precisión: {precision:.2f}", transform=plt.gca().transAxes, ha='left', va='top')
plt.savefig('C:/PythonResultado/kmeans.png')
plt.show()

# Aplicar kmedoids clustering
np.random.seed(77)  # Establecer una semilla aleatoria para reproducibilidad
kmedoids_instance = kmedoids(data_pca, initial_index_medoids=np.random.randint(0, data_pca.shape[0], num_clusters))
kmedoids_instance.process()
#se obtiene la clasisicacion como una lista de listas
cluster_labels_kmedoids = kmedoids_instance.get_clusters()
# Visualizar los resultados
#colors = ['g', 'y', 'b', 'r']
# Obtener la clasificación predicha en una sola lista, sin alterar el orden
clasificacion_predicha_o = list(itertools.chain.from_iterable(cluster_labels_kmedoids))
#dimension de clasificadores
dimension_kmedoids = len(clasificacion_predicha_o)
clasificacion_predicha_f = []
clasificacion_original_en_orden_de_kmedoids = [None] * dimension_kmedoids
for i, cluster in enumerate(cluster_labels_kmedoids): 
    #print(i)
    #print(cluster)
    #clasificacion_predicha.extend([i for _ in cluster])
    for point_index in cluster:
        clasificacion_original_en_orden_de_kmedoids[point_index]=i
        plt.scatter(data_pca[point_index, 0], data_pca[point_index, 1], color=colors[int(clasificacion_original_en_orden_de_kmedoids[point_index]) % len(colors)], alpha=0.5)
        #plt.scatter(data_pca[point_index, 0], data_pca[point_index, 1], color=colors[i % len(colors)], alpha=0.5)
plt.title('PCA + K-medoids Clustering')
# Calcular precisión
precision = accuracy_score(columna_respaldo, clasificacion_original_en_orden_de_kmedoids)
plt.text(0.05, 0.95, f"Precisión: {precision:.2f}", transform=plt.gca().transAxes, ha='left', va='top')
plt.savefig('C:/PythonResultado/kmedoids.png')
plt.show()

# Generar un conjunto de datos de ejemplo (en forma de dos lunas)
#X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
# Crear el objeto DBSCAN y ajustar el modelo a los datos
dbscan = DBSCAN(eps=0.5, min_samples=8)
dbscan.fit(data_pca)
# Obtener las etiquetas de los clústeres (-1 representa el ruido)
labelsDBSCAN = dbscan.labels_
# Asignar nuevos grupos a las etiquetas, reemplazando el ruido (-1) con el número de grupos + 1
labelsDBSCAN[labelsDBSCAN == -1] = num_clusters+1
# Imprimir los resultados
# Visualización de los resultados
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labelsDBSCAN)
plt.title("PCA + DBSCAN Clustering")
# Calcular precisión
precision = accuracy_score(columna_respaldo, labelsDBSCAN)
plt.text(0.05, 0.95, f"Precisión: {precision:.2f}", transform=plt.gca().transAxes, ha='left', va='top')
plt.savefig('C:/PythonResultado/DBSCAN.png')
plt.show()
