"""
Утилиты для визуализации и анализа.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_label_distribution(labels: pd.Series, save_path: str = None) -> None:
    """
    Построение графика распределения меток в датасете.
    
    Аргументы:
        labels: Series с метками.
        save_path: Опциональный путь для сохранения графика.
    """
    label_counts = labels.value_counts()
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=['Real', 'Fake'],
                y=label_counts,
                text=label_counts,
                textposition='auto',
            )
        ]
    )
    
    fig.update_layout(
        title='Распределение реальных и фейковых новостей',
        xaxis_title='Метка',
        yaxis_title='Количество',
        showlegend=False
    )
    
    if save_path:
        fig.write_image(save_path)
        print(f"График сохранен в {save_path}")
    else:
        fig.show()


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list = None,
    save_path: str = None
) -> None:
    """
    Построение матрицы ошибок для оценки модели.
    
    Аргументы:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.
        labels: Список меток классов.
        save_path: Опциональный путь для сохранения графика.
    """
    if labels is None:
        labels = ['FAKE', 'REAL']
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm_normalized,
        xticklabels=labels,
        yticklabels=labels,
        cmap='Blues',
        annot=True,
        fmt='.4f',
        square=True
    )
    
    plt.title('Матрица ошибок (нормализованная)', fontsize=16, pad=20)
    plt.ylabel('Истинные метки', fontsize=12)
    plt.xlabel('Предсказанные метки', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Матрица ошибок сохранена в {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_metrics(metrics: dict, save_path: str = None) -> None:
    """
    Построение графиков метрик обучения.
    
    Аргументы:
        metrics: Словарь с метриками.
        save_path: Опциональный путь для сохранения графика.
    """
    if 'accuracy' not in metrics:
        print("Метрика accuracy не найдена в словаре metrics")
        return
    
    accuracy = metrics['accuracy']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # График точности
    ax1.bar(['Accuracy'], [accuracy], color='skyblue')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Точность')
    ax1.set_title(f'Точность модели: {accuracy:.4f}')
    ax1.grid(axis='y', alpha=0.3)
    
    # Добавление значения над столбцом
    ax1.text(0, accuracy + 0.01, f'{accuracy:.4f}', 
             ha='center', va='bottom', fontweight='bold')
    
    # Матрица ошибок, если доступна
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        im = ax2.imshow(cm, cmap='Blues')
        ax2.set_title('Матрица ошибок')
        ax2.set_xlabel('Предсказано')
        ax2.set_ylabel('Истинно')
        
        # Добавление текстовых аннотаций
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, str(cm[i, j]), 
                        ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['FAKE', 'REAL'])
        ax2.set_yticklabels(['FAKE', 'REAL'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График метрик сохранен в {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_classification_report(metrics: dict) -> None:
    """
    Вывод форматированного отчета о классификации.
    
    Аргументы:
        metrics: Словарь с отчетом о классификации.
    """
    if 'classification_report' not in metrics:
        print("Отчет о классификации не найден в словаре metrics")
        return
    
    report = metrics['classification_report']
    
    print("\n" + "="*60)
    print("ОТЧЕТ О КЛАССИФИКАЦИИ")
    print("="*60)
    
    for label in ['FAKE', 'REAL']:
        if label in report:
            label_metrics = report[label]
            print(f"\n{label}:")
            print(f"  Точность:    {label_metrics['precision']:.4f}")
            print(f"  Полнота:     {label_metrics['recall']:.4f}")
            print(f"  F1-мера:     {label_metrics['f1-score']:.4f}")
            print(f"  Поддержка:   {label_metrics['support']}")
    
    if 'accuracy' in report:
        print(f"\nОбщая точность: {report['accuracy']:.4f}")
    
    print("="*60)