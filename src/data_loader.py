"""
Модуль загрузки и предобработки данных для обнаружения фейковых новостей.
"""

import pandas as pd
import requests
from typing import Tuple, Optional


class DataLoader:
    """
    Класс для загрузки и предобработки датасета новостей.
    
    Атрибуты:
        df (pd.DataFrame): Загруженный датасет.
        data_url (str): URL для скачивания датасета.
    """
    
    def __init__(self, data_url: Optional[str] = None):
        """
        Инициализация DataLoader.
        
        Аргументы:
            data_url: URL для скачивания датасета. Если None, используется URL по умолчанию.
        """
        self.df = None
        self.data_url = data_url or (
            "https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv"
        )
    
    def load_dataset(self, save_path: str = "data/raw/fake_news.csv") -> pd.DataFrame:
        """
        Скачивание и загрузка датасета.
        
        Аргументы:
            save_path: Путь для сохранения скачанного датасета.
            
        Возвращает:
            pd.DataFrame: Загруженный датасет.
        """
        print(f"Скачивание датасета из {self.data_url}...")
        
        # Создание директории, если она не существует
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Скачивание датасета
        response = requests.get(self.data_url)
        response.raise_for_status()
        
        # Сохранение датасета
        with open(save_path, 'wb') as file:
            file.write(response.content)
        
        print(f"Датасет сохранен в {save_path}")
        
        # Загрузка датасета
        self.df = pd.read_csv(save_path, sep=',')
        
        return self.df
    
    def preprocess(self) -> pd.DataFrame:
        """
        Предобработка датасета.
        
        Возвращает:
            pd.DataFrame: Предобработанный датасет.
        """
        if self.df is None:
            raise ValueError("Датасет не загружен. Сначала вызовите load_dataset().")
        
        # Переименование столбца
        self.df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        
        # Установка индекса
        self.df.set_index('id', inplace=True)
        
        print(f"Размер датасета: {self.df.shape}")
        print(f"Столбцы: {list(self.df.columns)}")
        
        # Проверка на пропущенные значения
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Обнаружены пропущенные значения:\n{missing_values}")
            # Заполнение пропущенных значений пустой строкой
            self.df.fillna('', inplace=True)
        
        # Отображение распределения меток
        print(f"\nРаспределение меток:")
        print(self.df.label.value_counts())
        
        return self.df
    
    def get_text_label_columns(self) -> Tuple[pd.Series, pd.Series]:
        """
        Получение столбцов с текстом и метками для обучения.
        
        Возвращает:
            Tuple[pd.Series, pd.Series]: Столбцы с текстом и метками.
        """
        if self.df is None:
            raise ValueError("Датасет не загружен. Сначала вызовите load_dataset().")
        
        return self.df['text'], self.df['label']