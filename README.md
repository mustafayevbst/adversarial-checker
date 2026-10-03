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
- Работать через CLI-инструмент `check.py`

## Быстрый старт

1. Клонируй репозиторий и перейди в папку:

```bash
git clone https://github.com/mustafayevbst/adversarial-checker.git
cd adversarial-checker
```

2. Создай venv и установи зависимости:

Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Обучи модели:

```bash
python train.py
python train_adv.py
```

4. Проверь устойчивость:

```bash
python check.py --model model.pth --attack fgsm
```

## Использование CLI

```bash
# Чистая точность (без атаки)
python check.py --model model.pth --attack none

# FGSM-атака
python check.py --model model.pth --attack fgsm --eps 0.3

# PGD-атака (40 шагов, alpha = eps/10)
python check.py --model model_adv.pth --attack pgd --eps 0.3 --steps 40

# Справка
python check.py --help
```

**Параметры:**
- `--eps` — максимальное возмущение пикселей (ε, обычно 0.3 для MNIST)
- `--steps` — число шагов для PGD
- `--attack` — `fgsm`, `pgd` или `none`

## Параметры экспериментов

- Датасет: MNIST (60 000 train, 10 000 test)
- Модель: простая CNN (2 свёртки + 2 пулинга + 2 линейных слоя)
- Optimizer: Adam, lr=1e-3
- Эпох: 5
- Batch size:
  - Vanilla: 64
  - Adversarial: 64 (64 чистых + 64 атакованных = 128 за шаг)
- Атаки для оценки: FGSM и PGD с `eps=0.3`, PGD — 40 шагов, `alpha=0.03`
- Adversarial training: FGSM с `eps=0.3`

## Результаты

| Модель | Clean accuracy | FGSM ASR (eps=0.3) | PGD ASR (eps=0.3, steps=40) |
|--------|----------------|---------------------|------------------------------|
| Vanilla (`model.pth`) | 99.01% | 81.70% | 100% |
| Adversarial (`model_adv.pth`) | 98.67% | 1.77% | 100% |

**Как считается ASR:** доля всех тестовых примеров, на которых модель дала неверное предсказание после атаки. Формула: `ASR = 1 - accuracy_after_attack`.

**Вывод:** adversarial training с FGSM резко снижает уязвимость к FGSM (81.7% → 1.77%), но оставляет модель уязвимой к PGD. Это ожидаемо: модель, обученная на одношаговой атаке, плохо защищена от итеративных. Возможные причины — overfitting на FGSM или gradient masking. Для защиты от PGD нужен adversarial training с PGD (Madry et al., 2018).

## Структура проекта

```
adversarial-checker/
├── dataset.py           # загрузка MNIST
├── model.py             # архитектура SimpleCNN
├── train.py             # обычное обучение
├── train_adv.py         # adversarial training
├── attack.py            # базовые атаки + визуализация
├── attack_compare.py    # сравнение двух моделей
├── check.py             # CLI-инструмент
├── tests/               # pytest-тесты
│   └── test_model.py
├── .github/workflows/   # CI (GitHub Actions)
│   └── ci.yml
├── requirements.txt
├── LICENSE
└── README.md
```

**Локально создаются (но не в репозитории):**
- `model.pth`, `model_adv.pth` — веса моделей (создаются через `train.py`)
- `data/` — датасет MNIST (скачивается автоматически)
- `attack_example.png` — визуализация атаки (создаётся через `attack.py`)
- `venv/` — виртуальное окружение

## Требования

- Python 3.9+
- См. `requirements.txt`

## TODO

- [ ] Adversarial training с PGD (Madry et al.) — для защиты от итеративных атак
- [ ] Добавить C&W и DeepFool
- [ ] Поддержка CIFAR-10
- [ ] Сохранение отчёта в JSON
- [ ] Оценка через black-box атаки (SimBA, HopSkipJump)

## Лицензия

MIT. См. файл `LICENSE`.