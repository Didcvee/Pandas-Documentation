import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
data = pd.read_csv('new_result.csv')

data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'store_and_fwd_flag', 'Month'], axis=1, inplace=True)

descriptions = {

}
def check_missing_values(col):
    missing_values = data[col].isnull().sum()
    print(f"Количество пустых значений в {col}: {missing_values}")

def plot_density(col):
    data[col].plot(kind='density')
    plt.title(f"Анализ плотности распределения значений {col}")
    plt.show()

def check_normality(col):
    _, p_value = stats.normaltest(data[col].dropna())
    alpha = 0.05
    if p_value < alpha:
        print(f"Атрибут {col} не имеет нормального распределения (p-value = {p_value})")
    else:
        print(f"Атрибут {col} имеет нормальное распределение (p-value = {p_value})")

def analyze_distribution(col):
    print(f"Описание атрибута {col}: {descriptions[col]}")
    check_missing_values(col)
    plot_density(col)
    check_normality(col)
    print("------------------------------------")

def aloha():
    # Проход по каждому атрибуту и проведение анализа
    for col in data.columns:
        analyze_distribution(col)
aloha()

# если плохая нормализация
numeric_columns = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge']

data[data < 0] = 0

for col in numeric_columns:
    data[col] = np.log1p(data[col])  # np.log1p применяет логарифмическое преобразование с добавлением 1 для обработки нулевых значений

aloha()