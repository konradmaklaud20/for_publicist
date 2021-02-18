import pandas as pd
import plotly
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px

df = pd.read_csv('rzn_ved2.csv')  # Датасет газеты

df = df.dropna().reset_index(drop=True)  # На всякий случай убираем все пустые ячейки
df = df.drop_duplicates().reset_index(drop=True)  # ... И все дубликаты

print(list(df['tag'].unique()))  # Показываем все уникальные рубрики

df = df['tag'].value_counts(sort=True)  # Сортируем рубрики по количеству встречаемости
df1 = pd.DataFrame(df)  # Создаём новый датафрейм для удобства
df1['tag_name'] = df1.index  # Создаём колонку с названием рубрики 
df1['tag'] = df1['tag'].apply(lambda x: int(x))  # Преобразуем значение количества встречаемости рубрики из строкового типа данных в целочисленное

df1.loc[df1['tag'] < 200, 'tag_name'] = 'Другие'  # Для наглядности возьмём только те рубрики, которые встретились более 200 раз

df1 = df1.rename(columns={'tag': 'count', 'tag_name': 'tag'})  # Переименовываем столбцы

fig = px.pie(df1, values='count', names='tag', title='<b>Распределение рубрик<b>',
             color_discrete_sequence=px.colors.sequential.Agsunset)  # Строим график, указывая источник данных, заголовок и цвет
             
fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=17)  # Изменяем показ элементов на: название рубрики + процент её встречаемости
fig.update(layout=dict(title=dict(x=0.5), titlefont=dict(size=20)))  # Делаем заголовок по центру графика и увеличиваем шрифт 
div = plotly.io.to_html(fig, full_html=False, config=dict(displayModeBar=False))  # Преобразуем график для показа на веб-странице
