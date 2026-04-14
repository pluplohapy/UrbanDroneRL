# Гайд по параметрам проекта

## 1. Environment Parameters (config.py)

### Размеры арены
```python
ARENA_SIZE_X = 15.0      # Ширина арены (метры)
ARENA_SIZE_Y = 15.0      # Длина арены (метры)
ARENA_HEIGHT = 5.0       # Высота арены (метры)
```
**Влияние:** Больше арена = сложнее задача, дольше эпизоды

### Условия успеха
```python
MIN_START_GOAL_DIST = 8.0   # Минимальное расстояние старт-цель
SUCCESS_DIST = 1.0          # Радиус достижения цели (метры)
MAX_STEPS = 750             # Максимум шагов в эпизоде
```
**Как менять:**
- `SUCCESS_DIST`: 0.5 = сложнее, 2.0 = легче
- `MAX_STEPS`: 500 = дрон торопится, 1000 = больше времени

### Скорости дрона
```python
VX_MAX = 1.5        # Макс скорость вперед/назад (м/с)
VY_MAX = 1.5        # Макс скорость влево/вправо (м/с)
VZ_MAX = 0.8        # Макс скорость вверх/вниз (м/с)
YAW_RATE_MAX = 0.8  # Макс скорость поворота (рад/с)
```
**Влияние:** Больше скорость = быстрее летает, но сложнее контроль

---

## 2. Sensor Parameters (config.py)

### Raycasts (лидар)
```python
N_RAYS = 16         # Количество лучей (8 гориз + 4 вверх + 4 вниз)
RAY_LENGTH = 5.0    # Дальность луча (метры)
```
**Как менять:**
- `RAY_LENGTH`: 3.0 = близорукий, 7.0 = дальнозоркий
- Больше лучей = лучше видит, но медленнее обучение

---

## 3. Reward Parameters (config.py)

### Терминальные награды
```python
REWARD_SUCCESS = 500.0   # Награда за достижение цели
REWARD_CRASH = -100.0    # Штраф за столкновение
```
**Влияние:** 
- Больше `REWARD_SUCCESS` = сильнее мотивация достичь цели
- Больше `REWARD_CRASH` (по модулю) = осторожнее летает

### Progress reward
```python
REWARD_PROGRESS_SCALE = 6.0  # Множитель за приближение к цели
```
**Формула:** `reward = 6.0 * (prev_dist - curr_dist)`
**Влияние:** Больше = сильнее мотивация лететь к цели

### Proximity penalty (штраф за близость к препятствиям)
```python
REWARD_PROXIMITY_THRESHOLD = 0.5  # Расстояние активации штрафа (м)
REWARD_PROXIMITY_SCALE = 8.0      # Сила штрафа
```
**Формула:** `penalty = 8.0 * (0.5 - min_dist)` если `min_dist < 0.5`
**Как менять:**
- `THRESHOLD`: 0.3 = штраф только вплотную, 1.5 = штраф издалека
- `SCALE`: 5.0 = слабый штраф, 20.0 = сильный штраф

### Другие награды
```python
REWARD_STEP_PENALTY = 0.002         # Штраф за каждый шаг
REWARD_EXPLORATION_BONUS = 0.5      # Награда за новую клетку
EXPLORATION_GRID_SIZE = 1.0         # Размер клетки (метры)
```

### Hardcoded rewards (nav_aviary.py:203-208)
```python
velocity_reward = 0.2 * max(0, velocity_towards_goal)  # Награда за полет к цели
proximity_bonus = 15.0 * np.exp(-curr_dist)            # Награда за близость к цели
timeout_penalty = -50.0                                 # Штраф за timeout (строка 371)
```

---

## 4. Obstacle Parameters (config.py)

### Stage 1 (статические препятствия)
```python
STAGE1_N_OBSTACLES = (8, 12)     # Случайное количество от 8 до 12
STAGE1_RADIUS = (0.25, 0.50)     # Радиус цилиндра (метры)
STAGE1_HEIGHT = (2.5, 5.0)       # Высота цилиндра (метры)
MIN_CLEARANCE = 1.0              # Мин расстояние от старта/цели
```
**Как менять:**
- `N_OBSTACLES`: (3, 6) = легко, (12, 20) = сложно
- `RADIUS`: (0.4, 0.6) = толстые препятствия
- `HEIGHT`: (1.5, 3.0) = можно пролететь сверху

---

## 5. Training Parameters (config.py)

### Базовые
```python
SEED = 42           # Random seed
N_ENVS = 4          # Количество параллельных сред
TOTAL_STEPS = 600_000  # Не используется (задается в train.py)
```
**Влияние:**
- `N_ENVS`: 2 = медленно, 8 = быстро (но нужна мощность CPU)

### PPO Hyperparameters
```python
PPO_PARAMS = {
    "learning_rate": 3e-4,      # Скорость обучения
    "n_steps": 1024,            # Шагов перед update
    "batch_size": 128,          # Размер батча
    "n_epochs": 5,              # Эпох на update
    "gamma": 0.99,              # Discount factor
    "gae_lambda": 0.95,         # GAE lambda
    "clip_range": 0.15,         # PPO clip range
    "ent_coef": 0.005,          # Entropy coefficient
    "vf_coef": 0.5,             # Value function coef
    "max_grad_norm": 0.5,       # Gradient clipping
    "policy": "MlpPolicy",      # Тип политики
}
```

**Ключевые для изменения:**

#### learning_rate (скорость обучения)
- `3e-4` (0.0003) - стандарт для Stage 0
- `1e-4` (0.0001) - для fine-tuning Stage 1
- Больше = быстрее учится, но нестабильнее

#### n_steps (шагов перед обновлением)
- `1024` - стандарт
- `2048` - больше данных, стабильнее
- Должно быть кратно `N_ENVS`

#### ent_coef (exploration)
- `0.005` - стандарт
- `0.01` - больше исследования
- `0.001` - меньше исследования

#### Архитектура сети (добавить в PPO_PARAMS)
```python
"policy_kwargs": dict(
    net_arch=[128, 128]  # Размер скрытых слоев
)
```
- `[64, 64]` - дефолт (маленькая сеть)
- `[128, 128]` - средняя сеть
- `[256, 128]` - большая сеть
- `[128, 128, 64]` - три слоя

---

## 6. Training Scripts

### train.py (Stage 0)
```python
total_timesteps=1500000  # 1.5M шагов
```

### train_stage1.py (Stage 1)
```python
total_timesteps=2400000  # 2.4M шагов (текущее значение)
reset_num_timesteps=False  # Продолжить счетчик шагов
```

---

## Рекомендации для экспериментов

### Для улучшения облета препятствий:
1. **Увеличить ray length**: `RAY_LENGTH = 7.0`
2. **Раньше штрафовать**: `REWARD_PROXIMITY_THRESHOLD = 1.5`
3. **Сильнее штрафовать**: `REWARD_PROXIMITY_SCALE = 15.0`
4. **Больше лучей**: добавить 8 горизонтальных (изменить raycasts.py)

### Для ускорения обучения:
1. **Больше сред**: `N_ENVS = 8`
2. **Меньше препятствий**: `STAGE1_N_OBSTACLES = (5, 8)`
3. **Больше n_steps**: `n_steps = 2048`

### Для более агрессивного поведения:
1. **Больше скорость**: `VX_MAX = 2.0, VZ_MAX = 1.2`
2. **Меньше MAX_STEPS**: `MAX_STEPS = 500`
3. **Больше step penalty**: `REWARD_STEP_PENALTY = 0.005`

### Для более осторожного поведения:
1. **Больше crash penalty**: `REWARD_CRASH = -200.0`
2. **Раньше proximity**: `REWARD_PROXIMITY_THRESHOLD = 2.0`
3. **Меньше скорость**: `VX_MAX = 1.0`

---

## 7. Physics & Control Parameters (nav_aviary.py:44-45)

### Частоты симуляции
```python
pyb_freq=240        # Частота физики PyBullet (Гц)
ctrl_freq=30        # Частота управления (Гц)
```
**Влияние:**
- `pyb_freq`: 240 = точная физика, 120 = быстрее но менее точно
- `ctrl_freq`: 30 = 30 действий/сек, 60 = более отзывчивый контроль
- Соотношение: `pyb_freq / ctrl_freq` должно быть целым числом

### Velocity control multiplier (nav_aviary.py:89)
```python
target_pos = state[0:3] + target_vel_world * self.CTRL_TIMESTEP * 3.0
```
**Множитель 3.0** - насколько агрессивно PID контроллер следует за целевой скоростью
- `2.0` = плавнее, медленнее реакция
- `4.0` = агрессивнее, быстрее реакция

---

## 8. Hardcoded Rewards (nav_aviary.py)

### Velocity reward (строка 203)
```python
velocity_reward = 0.2 * max(0, velocity_towards_goal)
```
**Влияние:** Награда за полет в направлении цели
- `0.1` = слабая мотивация
- `0.5` = сильная мотивация лететь к цели

### Proximity bonus (строка 207)
```python
proximity_bonus = 15.0 * np.exp(-curr_dist)
```
**Влияние:** Экспоненциальная награда за близость к цели
- `10.0` = слабее притяжение к цели
- `20.0` = сильнее притяжение к цели

### Timeout penalty (строка 371)
```python
reward -= 50.0  # Штраф за timeout без достижения цели
```
**Влияние:** Штраф если эпизод закончился по MAX_STEPS
- `20.0` = слабый штраф
- `100.0` = сильный штраф за медлительность

---

## 9. Observation Space (nav_aviary.py:114-181)

### Текущие features (29 total):
1. Goal in body frame (3) - направление к цели
2. Distance to goal (1) - расстояние до цели
3. Velocity in body frame (3) - скорость дрона
4. Normalized height (1) - высота
5. Yaw angle (1) - ориентация
6. Previous action (4) - предыдущее действие
7. Raycasts (16) - датчики препятствий

**Что можно добавить:**
- Time To Collision (TTC) для каждого луча
- История raycasts (temporal context)
- Angular velocity (угловая скорость)

---

## 10. Termination Conditions (nav_aviary.py:239-277)

### Success (строка 250)
```python
if dist_to_goal < config.SUCCESS_DIST:
    return True
```

### Crash (строка 254)
```python
contact_points = p.getContactPoints(bodyA=self.DRONE_IDS[0])
if len(contact_points) > 0:
    return True
```

### Timeout (строка 268)
```python
if self.control_step_counter >= config.MAX_STEPS:
    return True
```

### Out of bounds (строка 272-276)
```python
if (abs(drone_pos[0]) > ARENA_SIZE_X / 2 or
    abs(drone_pos[1]) > ARENA_SIZE_Y / 2 or
    drone_pos[2] < 0.1 or
    drone_pos[2] > ARENA_HEIGHT):
    return True
```

---

## 11. Пропущенные важные параметры

### Stage 2 (динамические препятствия) - не используется
```python
STAGE2_N_DYNAMIC = (3, 6)           # Количество движущихся препятствий
STAGE2_DYN_AMPLITUDE = (0.5, 1.5)  # Амплитуда движения (метры)
STAGE2_DYN_FREQUENCY = (0.3, 1.0)  # Частота движения (Гц)
```

### Curriculum learning - не используется
```python
CURRICULUM_WINDOW = 200          # Окно для расчета success rate
THRESHOLDS = {0: 0.90, 1: 0.75}  # Пороги для перехода между stage
```

---

## Быстрый чеклист изменений

**Файл config.py** - основные параметры
**Файл nav_aviary.py:203-208** - hardcoded rewards (velocity, proximity bonus)
**Файл nav_aviary.py:371** - timeout penalty
**Файл nav_aviary.py:89** - velocity control multiplier
**Файл train_stage1.py:134** - количество шагов обучения
**Файл raycasts.py** - конфигурация лучей (если менять N_RAYS)
