import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('new_result.csv')

df['year'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.year

years = df['year'].unique()
print('Годы в датасете:')
print(years)

for year in years:
    print(f'Год: {year}')

    df_year = df[df['year'] == year]

    df_filtered = df[(df['trip_distance'] <= 2) & (df['tip_amount'] >= 0.15 * df['total_amount'])]

    pickup_counts = df_filtered['pulocationid'].value_counts()

    top_pickup_locations = pickup_counts[pickup_counts == pickup_counts.max()]

    print('Топ районов с наибольшим количеством посадок и чаевыми более 15%:')
    print(top_pickup_locations)

    avg_fare_per_km = df.groupby('ratecodeid')['fare_amount'].mean() / df.groupby('ratecodeid')['trip_distance'].mean()

    print('Средняя стоимость поездки на километр по тарифу:')
    print(avg_fare_per_km)

    popular_ratecodes = df['ratecodeid'].value_counts().head()

    avg_passenger_count = df[df['ratecodeid'].isin(popular_ratecodes.index)]['passenger_count'].mean()

    print('Среднее количество пассажиров на поездку с наиболее популярными тарифами:')
    print(avg_passenger_count)

    print('\n')
