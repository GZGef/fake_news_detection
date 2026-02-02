"""
Модуль модели для обнаружения фейковых новостей с использованием TF-IDF и Passive Aggressive Classifier.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class FakeNewsDetector:
    """
    Модель для обнаружения фейковых новостей с использованием TF-IDF и Passive Aggressive Classifier.
    
    Атрибуты:
        tfidf_vectorizer (TfidfVectorizer): Векторизатор TF-IDF для преобразования текста.
        classifier (PassiveAggressiveClassifier): Классификатор.
        is_trained (bool): Обучена ли модель.
    """
    
    def __init__(self, max_iter: int = 50, random_state: int = 7):
        """
        Инициализация FakeNewsDetector.
        
        Аргументы:
            max_iter: Максимальное количество итераций для классификатора.
            random_state: Семя для воспроизводимости.
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=0.7
        )
        self.classifier = PassiveAggressiveClassifier(max_iter=max_iter)
        self.random_state = random_state
        self.is_trained = False
    
    def prepare_data(
        self,
        texts: pd.Series,
        labels: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения и тестирования.
        
        Аргументы:
            texts: Текстовые данные.
            labels: Метки.
            test_size: Доля данных для тестовой выборки.
            
        Возвращает:
            Tuple: x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state
        )
        
        print(f"Размер обучающей выборки: {len(x_train)}")
        print(f"Размер тестовой выборки: {len(x_test)}")
        
        return x_train, x_test, y_train, y_test
    
    def train(self, x_train: pd.Series, y_train: pd.Series) -> None:
        """
        Обучение модели.
        
        Аргументы:
            x_train: Обучающие тексты.
            y_train: Обучающие метки.
        """
        print("Обучение TF-IDF векторизатора...")
        tfidf_train = self.tfidf_vectorizer.fit_transform(x_train)
        
        print("Обучение классификатора...")
        self.classifier.fit(tfidf_train, y_train)
        
        self.is_trained = True
        print("Обучение завершено!")
    
    def predict(self, texts: pd.Series) -> np.ndarray:
        """
        Предсказание на новых данных.
        
        Аргументы:
            texts: Текстовые данные для предсказания.
            
        Возвращает:
            np.ndarray: Предсказанные метки.
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        tfidf_texts = self.tfidf_vectorizer.transform(texts)
        return self.classifier.predict(tfidf_texts)
    
    def evaluate(self, x_test: pd.Series, y_test: pd.Series) -> dict:
        """
        Оценка модели на тестовых данных.
        
        Аргументы:
            x_test: Тестовые тексты.
            y_test: Тестовые метки.
            
        Возвращает:
            dict: Метрики оценки.
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите train().")
        
        y_pred = self.predict(x_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        print(f"Точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nОтчет о классификации:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def predict_single(self, text: str) -> str:
        """
        Предсказание для одного текста.
        
        Аргументы:
            text: Текст для классификации.
            
        Возвращает:
            str: Предсказанная метка ('Ложь' или 'Правда').
        """
        prediction = self.predict(pd.Series([text]))[0]
        return 'Ложь' if prediction == 'FAKE' else 'Правда'