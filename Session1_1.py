import pandas as pd
import numpy as np

df = pd.read_csv('result.csv')

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

df['Month'] = df['tpep_pickup_datetime'].dt.month

counts_per_month = df['Month'].value_counts()

target_counts_per_month = (0.4 * len(df) * counts_per_month / sum(counts_per_month)).astype(int)

random_indices = []
for month in counts_per_month.index:
    indices = df[df['Month'] == month].index.to_list()
    random_indices.extend(np.random.choice(indices, target_counts_per_month[month], replace=False))

df_subset = df.loc[random_indices]
df_subset.to_csv('new_result.csv', index=False)