import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Вычисляем коэффициент силуэта для оценки качества кластеризации
silhouette_avg = silhouette_score(X_scaled, labels)

# Выводим коэффициент силуэта
print('Коэффициент силуэта:', silhouette_avg)

# Визуализируем кластеры (если возможно)
plt.scatter(X_scaled['passenger_count'], X_scaled['trip_distance'], c=labels)
plt.xlabel('Passenger Count')
plt.ylabel('Trip Distance')
plt.title('Clustering Result')
plt.show()