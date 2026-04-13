# RL Drone Navigation

Обучение квадрокоптера навигации от точки A до B с использованием PPO и PyBullet.

## Установка

```bash
conda create -n drone_nav python=3.10 -y
conda activate drone_nav
conda install -c conda-forge pybullet -y
pip install gymnasium "stable-baselines3[extra]" scipy control transforms3d
pip install --no-deps git+https://github.com/utiasDSL/gym-pybullet-drones.git
```

## Использование

```bash
# Обучение (300k шагов, ~5-7 минут)
python train.py

# Визуализация
python visualize.py --model models/ppo_drone_nav --normalize models/vec_normalize.pkl
```

## Структура

```
├── config.py           # Константы и гиперпараметры
├── train.py            # Скрипт обучения
├── visualize.py        # Визуализация
├── envs/
│   ├── nav_aviary.py  # Среда с velocity control
│   ├── raycasts.py    # 16-лучевой сенсор
│   └── obstacles.py   # Препятствия
└── scenarios/
    └── stage0_empty.py # Пустая сцена
```

## Результаты (300k шагов)

- **Success rate: 20%** (1 из 5 эпизодов достигает цели)
- **Средняя дистанция: 1.15м** (цель: < 0.5м)
- **Лучший результат: 0.50м** (точное попадание)
- **Награды: до +1467**

## Исправленные баги

### 1. Truncation bug
Эпизоды обрывались на 57 шагах из-за конфликта имён `step_counter` с BaseRLAviary.

### 2. SPEED_LIMIT bug
BaseRLAviary ограничивает скорость до 0.25 м/с. Переопределили `_preprocessAction()`.

### 3. Body→World transform
Действия в body frame передавались в PID как world frame. Добавили преобразование через rotation matrix.

## Параметры

**Среда:**
- Дрон: Crazyflie 2.X
- Арена: 15×15×5 м
- Лимит: 750 шагов (25 сек)
- Успех: дистанция < 0.5м

**Наблюдения (27D):**
- Цель в body frame (3)
- Скорость в body frame (3)
- Высота (1)
- Предыдущее действие (4)
- Raycasts (16)

**Действия (4D):**
- vx, vy ∈ [-1.5, 1.5] м/с
- vz ∈ [-0.8, 0.8] м/с
- yaw_rate ∈ [-0.8, 0.8] рад/с

**Награды:**
```python
reward = 6.0 × progress
       + 0.1 × velocity_towards_goal
       + 10.0 × exp(-distance)
       - 1.2 × proximity_penalty
       - 0.002 × step
       + 100.0 (success)
       - 100.0 (crash)
```
