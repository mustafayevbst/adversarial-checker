# Adversarial Robustness Checker

Пет-проект по безопасности машинного обучения: оценка устойчивости нейросетей к состязательным атакам.

## Что умеет

- Обучать CNN на MNIST
- Проверять устойчивость модели к атакам:
  - **FGSM** — одношаговая атака
  - **PGD** — многошаговая атака
- Считать **ASR** (Attack Success Rate) — долю примеров, на которых атака обманула модель
- Обучать устойчивую модель через **adversarial training**
- Сравнивать vanilla и adversarial модели
- Работать через CLI

## Быстрый старт

1. Клонируй репозиторий:

    git clone https://github.com/mustafayevbst/adversarial-checker.git
    cd adversarial-checker

2. Создай venv и установи зависимости:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

3. Обучи модели:

    python train.py
    python train_adv.py

4. Проверь устойчивость:

    python check.py --model model.pth --attack fgsm

## Использование CLI

    # Чистая точность (без атаки)
    python check.py --model model.pth --attack none

    # FGSM-атака
    python check.py --model model.pth --attack fgsm --eps 0.3

    # PGD-атака
    python check.py --model model_adv.pth --attack pgd --eps 0.3 --steps 40

    # Справка
    python check.py --help

## Результаты

| Модель | Clean accuracy | FGSM ASR | PGD ASR |
|--------|----------------|----------|---------|
| Vanilla (model.pth) | 99.01% | 81.70% | 100% |
| Adversarial (model_adv.pth) | 98.67% | 1.77% | 100% |

**Вывод:** adversarial training резко снижает уязвимость к FGSM (81.7% → 1.77%), но не защищает от PGD — это пример gradient masking.

## Структура проекта

    adversarial-checker/
    ├── dataset.py           # загрузка MNIST
    ├── model.py             # архитектура SimpleCNN
    ├── train.py             # обычное обучение
    ├── train_adv.py         # adversarial training
    ├── attack.py            # базовые атаки + визуализация
    ├── attack_compare.py    # сравнение двух моделей
    ├── check.py             # CLI-инструмент
    ├── requirements.txt
    └── README.md

## TODO

- [ ] Добавить C&W и DeepFool
- [ ] Adversarial training с PGD
- [ ] Поддержка CIFAR-10
- [ ] Сохранение отчета в JSON

## Лицензия

MIT