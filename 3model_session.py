import pandas as pd

# Загрузка данных из файла csv
data = pd.read_csv("data.csv")

import numpy as np

# Выбор нужных столбцов для кластеризации
selected_columns = ['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount']
X = data[selected_columns]

# Заполнение пропущенных значений
X = X.fillna(0)

# Преобразование DataFrame в массив numpy
X = np.array(X)


from sklearn.cluster import KMeans

# Создание модели KMeans
kmeans_model = KMeans(n_clusters=3, random_state=42)

# Обучение модели на данных
kmeans_model.fit(X)

# Получение меток кластеров
kmeans_labels = kmeans_model.labels_

# Добавление меток кластеров в исходные данные
data['kmeans_cluster'] = kmeans_labels


from sklearn.cluster import DBSCAN

# Создание модели DBSCAN
dbscan_model = DBSCAN(eps=0.3, min_samples=10)

# Обучение модели на данных
dbscan_model.fit(X)

# Получение меток кластеров
dbscan_labels = dbscan_model.labels_

# Добавление меток кластеров в исходные данные
data['dbscan_cluster'] = dbscan_labels


from sklearn.cluster import DBSCAN

# Создание модели DBSCAN
dbscan_model = DBSCAN(eps=0.3, min_samples=10)

# Обучение модели на данных
dbscan_model.fit(X)

# Получение меток кластеров
dbscan_labels = dbscan_model.labels_

# Добавление меток кластеров в исходные данные
data['dbscan_cluster'] = dbscan_labels


from sklearn.metrics import silhouette_score

# Вычисление значения силуэта для каждого алгоритма
kmeans_silhouette_score = silhouette_score(X, kmeans_labels)
dbscan_silhouette_score = silhouette_score(X, dbscan_labels)
agg_silhouette_score = silhouette_score(X, agg_clustering_labels)

print("Silhouette Score (K-means):", kmeans_silhouette_score)
print("Silhouette Score (DBSCAN):", dbscan_silhouette_score)
print("Silhouette Score (Agglomerative):", agg_silhouette_score)


import matplotlib.pyplot as plt

# Визуализация результатов K-means
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title("K-means Clustering")
plt.xlabel("passenger_count")
plt.ylabel("trip_distance")
plt.show()

# Визуализация результатов DBSCAN
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels)
plt.title("DBSCAN Clustering")
plt.xlabel("passenger_count")
plt.ylabel("trip_distance")
plt.show()

# Визуализация результатов Agglomerative Clustering
plt.scatter(X[:, 0], X[:, 1], c=agg_clustering_labels)
plt.title("Agglomerative Clustering")
plt.xlabel("passenger_count")
plt.ylabel("trip_distance")
plt.show()
