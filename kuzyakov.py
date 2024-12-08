import pandas as pd
import seaborn as sns
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime

# Загрузка данных
data0 = pd.read_csv('C:/PProjects/crimes/Chicago_Crimes_2001_to_2004.csv', on_bad_lines="skip", low_memory=False)
data1 = pd.read_csv('C:/PProjects/crimes/Chicago_Crimes_2005_to_2007.csv', on_bad_lines="skip", low_memory=False)
data2 = pd.read_csv('C:/PProjects/crimes/Chicago_Crimes_2008_to_2011.csv', on_bad_lines="skip", low_memory=False)
data3 = pd.read_csv('C:/PProjects/crimes/Chicago_Crimes_2012_to_2017.csv', on_bad_lines="skip", low_memory=False)

# Объединение данных
data = pd.concat([data0, data1, data2, data3], ignore_index=True)

# Очистка и проверка данных
print(data.head())
print('Количество строк до чистки: ',len(data))
data = data.dropna() # Удаляем строки с пропущенными значениями
print('Строк после удаления пропущенных значений: ', data[data.columns[0]].count()) 
data.drop_duplicates(inplace=True) # Удаляем строки-дубликаты
print('Строк, после удаления дубликатов: ', data[data.columns[0]].count()) 

# Преобразование столбца 'Date' в формат datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Добавление столбцов года, месяца и времени суток
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Hour'] = data['Date'].dt.hour
data['Time_of_Day'] = pd.cut(
    data['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False
)

print(data.head())

# Фильтрация данных за последние три года
last_year = data['Year'].max()
data_filtered = data[data['Year'].isin([last_year, last_year - 1, last_year - 2])]

# Подготовка данных для визуализации на карте
crime_locations = data[['Latitude', 'Longitude', 'Primary Type']]

# Убедимся, что Latitude и Longitude имеют только числовые значения
data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

# Удаляем строки с пропущенными или некорректными значениями координат
data = data.dropna(subset=['Latitude', 'Longitude'])

# Проверяем данные
if data['Latitude'].empty or data['Longitude'].empty:
    raise ValueError("Нет данных для создания карты, проверьте исходный датасет.")

# Преобразуем Latitude и Longitude в числовой формат, обрабатываем ошибки
data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

# Удаляем строки с NaN после преобразования
data = data.dropna(subset=['Latitude', 'Longitude'])

# Проверяем на пустые данные
if data['Latitude'].empty or data['Longitude'].empty:
    raise ValueError("Данные для координат отсутствуют после обработки. Проверьте исходный датасет.")

# Создание оптимизированной карты
def Optimized_map():
    crime_locations = data[['Latitude', 'Longitude', 'Primary Type']].dropna().sample(n=10000, random_state=42)

    # Карта с увеличенным масштабом и более чёткими тайлами
    crime_map = folium.Map(
        location=[crime_locations['Latitude'].mean(), crime_locations['Longitude'].mean()],
        zoom_start=13,
        tiles="CartoDB Positron",
        attr="Map tiles by CartoDB, CC BY 3.0 — Map data © OpenStreetMap contributors"
    )

    # Тепловая карта
    heat_data = crime_locations[['Latitude', 'Longitude']].values.tolist()
    HeatMap(heat_data, radius=3, min_opacity=0.5, blur=5).add_to(crime_map)

    # Маркеры с уменьшенным радиусом
    from folium.plugins import MarkerCluster
    marker_cluster = MarkerCluster().add_to(crime_map)
    for idx, row in crime_locations.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            popup=row['Primary Type'],
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(marker_cluster)

    # Сохранение карты
    crime_map.save('optimized_chicago_crime_map_clear.html')

# Вызов функции
Optimized_map()

# Фильтрация по популярным блокам
block_counts = data_filtered['Block'].value_counts()
threshold = 100  # Только блоки с более чем 100 наблюдениями
frequent_blocks = block_counts[block_counts > threshold].index
data_filtered = data_filtered[data_filtered['Block'].isin(frequent_blocks)]

# Группировка по параметрам и подсчет количества преступлений
crime_counts = data_filtered.groupby(['Year', 'Month', 'Block', 'Time_of_Day'], observed=False).size().reset_index(name='Crime Count')

# Преобразование категориальных переменных в дамми-признаки
crime_counts = pd.get_dummies(crime_counts, columns=['Block', 'Time_of_Day'], drop_first=True)

# Создание циклических признаков для месяца
crime_counts['Month_sin'] = np.sin(2 * np.pi * crime_counts['Month'] / 12)
crime_counts['Month_cos'] = np.cos(2 * np.pi * crime_counts['Month'] / 12)

# Разделение на признаки (X) и целевую переменную (y)
X = crime_counts.drop(['Crime Count'], axis=1)
y = crime_counts['Crime Count']

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Линейная регрессия
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr:.2f}, R^2: {r2_lr:.2f}")

# Случайный лес
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest MSE: {mse_rf:.2f}, R^2: {r2_rf:.2f}")

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=10,  # Количество случайных комбинаций
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_

y_pred_rf = best_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Optimized Random Forest MSE: {mse_rf:.2f}, R^2: {r2_rf:.2f}")
print(f"Best parameters: {random_search.best_params_}")

# Визуализация количества преступлений по времени суток
def crimes():
    crime_by_time_of_day = data_filtered.groupby(['Time_of_Day']).size().reset_index(name='Crime Count')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=crime_by_time_of_day, x='Time_of_Day', y='Crime Count', hue='Time_of_Day', palette='tab10', legend=False)
    plt.title('Количество преступлений по времени суток')
    plt.xlabel('Время суток')
    plt.ylabel('Количество преступлений')
    plt.show()

crimes()

# Визуализация предсказаний линейной регрессии
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='blue', label='Linear Regression Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Prediction')
plt.title('Linear Regression Predictions vs. Actual')
plt.xlabel('Actual Crime Count')
plt.ylabel('Predicted Crime Count')
plt.legend()
plt.show()

# Визуализация предсказаний случайного леса
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='green', label='Random Forest Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Prediction')
plt.title('Random Forest Predictions vs. Actual')
plt.xlabel('Actual Crime Count')
plt.ylabel('Predicted Crime Count')
plt.legend()
plt.show()

# Визуализация предсказаний оптимизированного случайного леса
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='orange', label='Optimized Random Forest Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Prediction')
plt.title('Optimized Random Forest Predictions vs. Actual')
plt.xlabel('Actual Crime Count')
plt.ylabel('Predicted Crime Count')
plt.legend()
plt.show()

# Сравнение моделей
models = ['Linear Regression', 'Random Forest', 'Optimized RF']
mse_values = [3.39, 2.19, 1.86]
r2_values = [0.30, 0.55, 0.62]

plt.figure(figsize=(12, 6))

# MSE comparison
plt.subplot(1, 2, 1)
plt.bar(models, mse_values, color=['gray', 'blue', 'green'])
plt.title('MSE Comparison')
plt.ylabel('MSE')

# R^2 comparison
plt.subplot(1, 2, 2)
plt.bar(models, r2_values, color=['gray', 'blue', 'green'])
plt.title('R^2 Comparison')
plt.ylabel('R^2')

plt.tight_layout()
plt.show()