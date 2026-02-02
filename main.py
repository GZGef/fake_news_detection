"""
Основной скрипт для обучения и оценки модели обнаружения фейковых новостей.

Этот скрипт демонстрирует полный рабочий процесс:
1. Загрузка и предобработка датасета
2. Обучение модели
3. Оценка модели
4. Предсказание на новых текстах
"""

import os
from src import DataLoader, FakeNewsDetector
from src.utils import (
    plot_label_distribution,
    plot_confusion_matrix,
    plot_training_metrics,
    print_classification_report,
)


def main():
    """Основная функция для запуска рабочего процесса обнаружения фейковых новостей."""
    
    print("="*70)
    print("ОБНАРУЖЕНИЕ ФЕЙКОВЫХ НОВОСТЕЙ - ПРОЦЕСС ОБУЧЕНИЯ МОДЕЛИ")
    print("="*70)
    
    # Шаг 1: Загрузка и предобработка данных
    print("\n[Шаг 1] Загрузка и предобработка датасета...")
    data_loader = DataLoader()
    data_loader.load_dataset()
    data_loader.preprocess()
    
    # Получение текста и меток
    texts, labels = data_loader.get_text_label_columns()
    
    # Шаг 2: Визуализация распределения меток
    print("\n[Шаг 2] Визуализация распределения меток...")
    plot_label_distribution(labels, save_path="reports/figures/label_distribution.png")
    
    # Шаг 3: Инициализация и подготовка модели
    print("\n[Шаг 3] Инициализация модели...")
    model = FakeNewsDetector(max_iter=50, random_state=7)
    
    # Подготовка данных для обучения
    x_train, x_test, y_train, y_test = model.prepare_data(texts, labels, test_size=0.2)
    
    # Шаг 4: Обучение модели
    print("\n[Шаг 4] Обучение модели...")
    model.train(x_train, y_train)
    
    # Шаг 5: Оценка модели
    print("\n[Шаг 5] Оценка модели...")
    metrics = model.evaluate(x_test, y_test)
    
    # Шаг 6: Визуализация результатов
    print("\n[Шаг 6] Визуализация результатов...")
    
    # Матрица ошибок
    plot_confusion_matrix(
        y_test,
        model.predict(x_test),
        labels=['FAKE', 'REAL'],
        save_path="reports/figures/confusion_matrix.png"
    )
    
    # Метрики обучения
    plot_training_metrics(metrics, save_path="reports/figures/metrics.png")
    
    # Форматированный отчет
    print_classification_report(metrics)
    
    # Шаг 7: Тестирование на новых данных
    print("\n[Шаг 7] Тестирование на новых данных...")
    
    test_texts = [
        """
        'Это действительно важный вопрос, говорит Ламберт.
        "Я не хочу передавать двум или трем людям," говорит она.
        "Я хочу, чтобы один человек был на связи." Возможно, есть конкретные контактные лица
        для разных областей, добавляет она, например, директор по сестринскому делу для связанных вопросов.
        Однако, "Я хочу знать, что могу зайти в кабинет исполнительного директора в любое время,
        задать любой вопрос и подать любую жалобу," подчеркивает она.
        "Я хочу знать, что этот человек доступен.
        Потому что иногда приходится подниматься до этого уровня."
        """,
        """
        Сенсационные новости: Ученые обнаружили, что Земля на самом деле плоская!
        Это революционное открытие переворачивает вековой научный консенсус.
        """,
        """
        Фондовый рынок достиг исторического максимума сегодня, поскольку экономические показатели
        показали сильный рост и позитивные данные по занятости.
        """
    ]
    
    print("\nПредсказания на тестовых текстах:")
    print("-" * 50)
    for i, text in enumerate(test_texts, 1):
        prediction = model.predict_single(text)
        print(f"Текст {i}: {prediction}")
        print(f"  Предпросмотр: {text[:100]}...")
        print()
    
    print("="*70)
    print("РАБОЧИЙ ПРОЦЕСС ЗАВЕРШЕН!")
    print("="*70)
    print("\nСгенерированные файлы:")
    print("  - data/raw/fake_news.csv (датасет)")
    print("  - reports/figures/label_distribution.png")
    print("  - reports/figures/confusion_matrix.png")
    print("  - reports/figures/metrics.png")
    print("\nТеперь можно использовать модель для предсказаний!")


if __name__ == "__main__":
    # Создание необходимых директорий
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    main()