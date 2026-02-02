"""
Пакет для обнаружения фейковых новостей

Машинное обучение для классификации новостей с использованием TF-IDF и Passive Aggressive Classifier.
"""

__version__ = "1.0.0"
__author__ = "Тимофей Крылов"
__email__ = "timofey.krylov.0206@gmail.com"

from .data_loader import DataLoader
from .model import FakeNewsDetector
from .utils import (
    plot_confusion_matrix,
    plot_label_distribution,
    plot_training_metrics,
    print_classification_report,
)

__all__ = [
    "DataLoader",
    "FakeNewsDetector",
    "plot_confusion_matrix",
    "plot_label_distribution",
    "plot_training_metrics",
    "print_classification_report",
]