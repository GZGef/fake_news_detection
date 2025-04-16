"""# Импортируем библиотеки"""

from requests import get
import pandas as pd
import numpy as np

from plotly import graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns

import tempfile
import os
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

"""### Загружаем датасет"""

# Создаем запрос и получаем с него файл
response = get("https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv")

# Открываем в режиме записи байтов
with open("fake_news.csv", 'wb') as file:
    file.write(response.content)

# Создаем объект DataFrame
df = pd.read_csv("fake_news.csv", sep=',')
df.info()

df.head()

# Переименовываем столбец "Unnamed: 0" в "id"
df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

# Преобразуем столбец 'id' в индекс
df.set_index('id', inplace=True)
df.head()

# Получим последние пять строк набора данных
df.tail()

"""### Проверка на наличие нулевых или пропущенных значений"""

df.isnull().sum()

df.label.value_counts()

"""### Построение столбчатой диаграммы"""

i = df.label.value_counts()
fig = go.Figure(
    data = [
        go.Bar(
            x = ['Real', 'Fake'],
            y = i,
            text = i,
            textposition = 'auto',
        )
    ]
)

fig.show()

"""### Разделение данных для обучения и тестирования"""

x_train, x_test, y_train, y_test = train_test_split(
    df['text'],
    df.label,
    test_size = 0.2, # Доля данных, которая будет выделена для тестовой выборки
    random_state = 7 # Начальное состояние генератора случайных чисел
)

print(f'Размерность x_train {x_train.shape}')
print(f'Размерность y_train {y_train.shape}')

"""### Инициализируем TfidVectorizer"""

tfidf_vectorizer = TfidfVectorizer(
    stop_words = 'english', # игнорировать стоп-слова английского языка
    max_df = 0.7
)

# Инициализируем тренировочные данные для модели на обучающих данных и создаем TF-IDF матрицы для этих данных
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
# Инициализируем тестовые данные для модели и создаем соответствующую TF-IDF матрицу для них
tfidf_test = tfidf_vectorizer.transform(x_test)

"""### Инициализуем PassiveAggressiveClassifier и обучаем модель"""

pac = PassiveAggressiveClassifier(max_iter=50)

# Обучаем модель при помощи алгоритма PAC
pac.fit(
    tfidf_train,
    y_train
)

"""### Прогнозируем на основе тестовых данных"""

y_pred = pac.predict(tfidf_test)

# Рассчитываем точность
score = accuracy_score(y_test, y_pred)
print(f'Точность: {round(score*100, 2)}%')

"""### Создаем матрицу ошибок"""

def show_confusion_matrix(y_test, y_pred):
    # Вычисление матрицы ошибок и ее нормализация
    plt.figure(figsize = (10, 10))
    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels = ['FAKE','REAL']
    )

    confusion_normalized = confusion/confusion.sum(axis=1, keepdims=True)
    axis_labels = ['FAKE','REAL']

    ax = sns.heatmap(
        confusion_normalized,
        xticklabels = axis_labels,
        yticklabels = axis_labels,
        cmap = 'Blues',
        annot = True,
        fmt = '.4f',
        square = True
    )
    plt.title("Матрица ошибок")
    plt.ylabel("Истинные метки")
    plt.xlabel("Предсказанные метки")

show_confusion_matrix(y_test, y_pred)

"""### Создаем отчет о классификации"""

print('\n Отчет о классификации:\n', classification_report(y_test, y_pred))

"""### Классифицируем на новых данных (не из набора)"""

TEST_TEXT = [
    '''
    'This is a really important question, Lambert says.
    “I don’t want to be passed along to two or three people,” she says.
    “I want one person to contact.” There may be specific contact points
    for different areas, she adds, such as the director of nursing for related questions.
    However, “I want to know that I can pop into the executive director’s office anytime,
    ask any question and make any kind of complaint,” she emphasizes.
    “I want to know that person is available.
    Because sometimes, you have to go up to that level.""'
    '''
]

tfidf_test = tfidf_vectorizer.transform(TEST_TEXT)
y_pred = pac.predict(tfidf_test)
print(f"Этот текст - {'Ложь' if y_pred == 'FAKE' else 'Правда'}")
