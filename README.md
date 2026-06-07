# Навигация дрона с обучением с подкреплением

Система автономной навигации дрона с использованием PPO/RecurrentPPO в симуляции PyBullet. Агент обучается управлять квадрокоптером Crazyflie от старта до цели в 3D-окружениях со статическими и динамическими препятствиями.

## Возможности

- **PPO и RecurrentPPO** обучение с Stable-Baselines3
- **Интеграция RRT*** для планирования пути через waypoints
- **Curriculum learning** на разных типах препятствий
- **Swarm режим** для нескольких дронов в общей среде
- **Safety shield** для избежания столкновений во время инференса
- Комплексные инструменты диагностики и оценки

## Быстрый старт

**Установка зависимостей:**

```bash
conda create -n drone_nav python=3.10
conda activate drone_nav
pip install -r requirements.txt
pip install sb3-contrib  # Опционально, для RecurrentPPO
```

**Обучение базовой модели:**

```bash
python training/train.py --stage 0 --timesteps 500000
```

**Визуализация обученной политики:**

```bash
python visualization/visualize.py \
  --model models/ppo_drone_nav_stage0_planner.zip \
  --normalize models/vec_normalize_stage0_planner.pkl \
  --stage 0
```

## Этапы обучения

| Этап | Описание | Карта | Планировщик |
|------|----------|-------|-------------|
| `0` | Базовая навигация в пустой арене | Пустое пространство 15×15×5м | RRT* включен |
| `1` | Навигация со статическими препятствиями | 8-12 цилиндров | RRT* включен |
| `pretrain` | Обучение локальному обходу препятствий | Коридоры, стены, динамические препятствия | Отключен |

## Структура проекта

```
.
├── config/           # Конфигурации обучения (stage 0/1/pretrain, debug)
├── envs/             # PyBullet Gymnasium окружения, raycast сенсор
├── planners/         # RRT* планировщик пути и утилиты waypoint
├── scenarios/        # Генераторы карт (пустая, статическая, pretrain/curriculum)
├── training/         # Скрипты обучения, оценки, curriculum
├── visualization/    # Инструменты GUI визуализации и рендеринга
├── tests/            # Юнит-тесты для окружения и планировщика
├── models/           # Сохраненные чекпоинты моделей
└── logs/             # TensorBoard логи и диагностика
```

## Примеры обучения

**Stage 0 (пустая арена с RRT*):**

```bash
python training/train.py --stage 0 --timesteps 500000 --eval
```

**Stage 1 (препятствия с transfer learning):**

```bash
python training/train.py --stage 1 --timesteps 1500000 --eval
```

**Pretrain на цилиндрах (без планировщика):**

```bash
python training/train.py \
  --stage pretrain \
  --obstacle-type cylinders \
  --timesteps 1000000 \
  --eval
```

**RecurrentPPO с динамическими препятствиями:**

```bash
python training/train.py \
  --algo recurrent_ppo \
  --stage pretrain \
  --obstacle-type dynamic_mix \
  --timesteps 1000000 \
  --eval
```

**Просмотр обучения в GUI:**

```bash
python training/train.py \
  --stage pretrain \
  --obstacle-type cylinders \
  --watch \
  --watch-fps 60
```

## Основные параметры

| Параметр | Описание |
|----------|----------|
| `--stage {0,1,pretrain}` | Этап/сложность обучения |
| `--algo {ppo,recurrent_ppo}` | Выбор алгоритма |
| `--obstacle-type TYPE` | Тип карты для pretrain этапа |
| `--no-planner` | Отключить RRT* планировщик |
| `--timesteps N` | Бюджет обучения |
| `--n-envs N` | Параллельные окружения |
| `--eval` | Включить периодическую оценку |
| `--watch` | GUI режим с визуализацией |
| `--swarm-drones N` | Несколько дронов на общей карте |

## Оценка модели

```bash
python training/evaluate.py \
  --model models/ppo_pretrain_cylinders_enhanced_obs.zip \
  --normalize models/vec_normalize_pretrain_cylinders_enhanced_obs.pkl \
  --stage pretrain \
  --obstacle-type cylinders \
  --episodes 100
```

## Визуализация

```bash
python visualization/visualize.py \
  --model models/ppo_pretrain_cylinders_enhanced_obs.zip \
  --normalize models/vec_normalize_pretrain_cylinders_enhanced_obs.pkl \
  --stage pretrain \
  --obstacle-type cylinders
```

## Тестирование

```bash
pytest tests/
```

## Технологический стек

- **PyBullet** - Физическая симуляция
- **gym-pybullet-drones** - Динамика Crazyflie
- **Stable-Baselines3** - Реализация PPO
- **sb3-contrib** - RecurrentPPO с LSTM
- **NumPy, SciPy** - Планирование пути и математика
- **Matplotlib** - Визуализация и графики

## Конфигурация

Основные конфигурационные файлы в `config/`:

- `base.py` - Базовые параметры (частоты, скорости, награды, PPO)
- `stage0.py` - Настройки пустой арены
- `stage1.py` - Настройки статических препятствий
- `pretrain.py` - Расширенные наблюдения, коридорные карты
- `debug.py` - Опции детального логирования

## Важные замечания

- Pretrain этап всегда работает без RRT* планировщика
- Stage 1 автоматически использует чекпоинт Stage 0 для transfer learning, если доступен
- Режим `--watch` принудительно использует одно окружение и фиксированную карту
- Всегда загружайте соответствующий `.pkl` файл нормализации с обученными моделями
- Расширенные наблюдения (`USE_ENHANCED_OBS`) изменяют пространство наблюдений

## Лицензия

Академический исследовательский проект.
