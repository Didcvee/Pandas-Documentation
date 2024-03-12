import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('result.csv')


# Выбираем признаки для кластеризации
features = ['passenger_count', 'trip_distance', 'fare_amount']

# Разделяем данные на обучающую и тестовую выборки (если нужно)
X = df[features]

# Стандартизируем признаки (если нужно)
X_scaled = (X - X.mean()) / X.std()

# Инициализируем модель кластеризации (например, KMeans с заданным количеством кластеров)
kmeans = KMeans(n_clusters=4, random_state=42)

# Обучаем модель на данных
kmeans.fit(X_scaled)

# Получаем метки кластеров для каждой записи
labels = kmeans.labels_

# Добавляем метки кластеров в исходный датасет
df['cluster'] = labels

# Выводим описание каждого кластера
clusters = df.groupby('cluster')[features].mean()
print('Описание кластеров:')
print(clusters)