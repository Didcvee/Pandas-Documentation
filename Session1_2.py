import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('new_result.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

le = LabelEncoder()

data['store_and_fwd_flag'] = le.fit_transform(data['store_and_fwd_flag'])

data['store_and_fwd_flag'] = data['store_and_fwd_flag'].astype(float)

correlation_matrix = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

significant_attributes = correlation_matrix.index[abs(correlation_matrix['total_amount']) > 0.5].tolist()

print("Наиболее значимые атрибуты:")
print(significant_attributes)