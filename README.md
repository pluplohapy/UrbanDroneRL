# Drone Navigation RL

Проект для обучения навигации квадрокоптера Crazyflie в 3D-среде PyBullet. Агент получает числовые наблюдения о цели, собственной скорости, границах арены и препятствиях, затем выбирает команду скорости `[vx, vy, vz, yaw_rate]`. Цель - долететь из стартовой точки к финишу без столкновений и выхода за границы.

Основной фокус репозитория:

- обучение PPO и RecurrentPPO для навигации дрона;
- сравнение прямого RL-управления с waypoint-навигацией через RRT*;
- pretrain/curriculum на статических и динамических картах;
- диагностика неудачных эпизодов, оценка моделей и генерация графиков для отчета.

## Технологии

- `gym-pybullet-drones` и `PyBullet` - физика Crazyflie, столкновения, GUI и raycast-сенсор;
- `stable-baselines3` - PPO, векторные среды, `VecNormalize`, callbacks;
- `sb3-contrib` - опциональный `RecurrentPPO` с LSTM;
- `numpy`, `scipy`, `matplotlib`, `pandas` - планирование, аналитика и графики;
- `pytest` - тесты среды, reward/observation logic и RRT*.

## Структура проекта

```text
.
├── config/                    # Конфиги stage 0, stage 1, pretrain и debug-настройки
├── envs/                      # Gymnasium/PyBullet среды, raycast-сенсор, препятствия
├── planners/                  # RRT* планировщик, waypoint utils, визуализация пути
├── scenarios/                 # Генераторы карт: пустая, статическая, pretrain/curriculum
├── training/                  # Обучение, оценка, curriculum-runner, построение графиков
├── visualization/             # GUI-просмотр, preview карт, рендеры для презентации
├── tests/                     # Автотесты RRT*, наблюдений, reward и runtime config sync
├── models/                    # Сохраненные модели и VecNormalize stats
├── logs/                      # TensorBoard, eval history, структурная диагностика
├── reports/                   # JSON-оценки, графики, изображения карт и аудит
├── COMMANDS.md                # Историческая шпаргалка по командам
├── PARAMETERS_GUIDE.md        # Исторический гайд по параметрам
└── PROJECT_CODE_DESCRIPTION.md # Подробное описание архитектуры и терминов
```

Файлы `models/`, `logs/` и `reports/` могут занимать много места. Это рабочие артефакты экспериментов, а не обязательная часть исходного кода.

## Режимы обучения

| Stage | Назначение | Карта | Планировщик |
| --- | --- | --- | --- |
| `0` | Базовая навигация к цели | пустая арена `15x15x5` м | RRT* включен по умолчанию |
| `1` | Навигация среди статических препятствий | `8-12` цилиндров | RRT* включен по умолчанию |
| `pretrain` | Обучение локального обхода на разных типах карт | коридорные, статические и динамические карты | RRT* отключен всегда |

Для `stage 1` обучение автоматически пытается использовать checkpoint `stage 0` как transfer learning, если доступны `models/ppo_drone_nav_stage0_planner.zip` и `models/vec_normalize_stage0_planner.pkl`.

Pretrain поддерживает типы карт:

```text
random, dynamic_mix, empty, cylinders, spheres, crossing_spheres,
walls, beams, boxes, gates, slalom, city_blocks, city_dynamic,
construction_site_dynamic, swinging_sticks
```

`city_dynamic` и `construction_site_dynamic` имеют собственные увеличенные размеры арены и дальность raycast, поэтому их лучше запускать как явный `--obstacle-type`, а не через случайную смесь.

В текущем pretrain-конфиге направление коридора фиксировано из `START_ZONE_Y` в `GOAL_ZONE_Y` (`PRETRAIN_BIDIRECTIONAL_GOALS=False`), а штраф за близость к границам включен (`REWARD_BOUNDARY_THRESHOLD=1.4`).

## Как устроен runtime

1. `Scenario` генерирует старт, цель и препятствия.
2. `NavAviary` поднимает PyBullet-среду, создает дрон, физические стены и raycast-сенсор.
3. Политика получает observation и возвращает action в диапазоне `[-1, 1]`.
4. Среда масштабирует action в реальные скорости, переводит их через PID Crazyflie в RPM моторов.
5. На каждом шаге считаются reward, collision/out-of-bounds/success flags, debug-метрики и terminal reward.
6. `Monitor`, TensorBoard, eval callback и `TrainingDiagnosticsLogger` сохраняют метрики для анализа.

### Action space

Один дрон:

```text
[vx, vy, vz, yaw_rate]
```

Все значения нормализованы в `[-1, 1]`. Лимиты скоростей берутся из активного конфига:

- base/stage 0/stage 1: `VX_MAX=1.2`, `VY_MAX=1.2`, `VZ_MAX=0.8`, `YAW_RATE_MAX=0.8`;
- pretrain: `VX_MAX=2.4`, `VY_MAX=2.4`, `VZ_MAX=1.6`, `YAW_RATE_MAX=1.2`.

Для swarm-режима action имеет форму `(num_drones, 4)`.

### Observation space

Базовое наблюдение содержит `33` признака:

- цель в body frame: `3`;
- нормированное расстояние до цели: `1`;
- скорость в body frame: `3`;
- высота: `1`;
- yaw: `1`;
- предыдущее действие: `4`;
- raycast-сенсор: `20`.

В pretrain включен `USE_ENHANCED_OBS`, поэтому добавляются:

- radial/lateral/braking признаки у цели: `3`;
- нормированное расстояние до ближайшей границы: `1`;
- изменения по каждому raycast-лучу: `20`.

Итого pretrain observation: `57` признаков.

## Установка

Рекомендуется отдельное окружение. Скрипт `quickstart.sh` ожидает conda-окружение `drone_nav`, но можно использовать любое окружение Python.

```bash
conda create -n drone_nav python=3.10
conda activate drone_nav
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Для RecurrentPPO:

```bash
python -m pip install sb3-contrib
```

Для тестов:

```bash
python -m pip install pytest
```

Если на macOS или в conda появляется OpenMP/KMP ошибка, можно запустить команды с:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Быстрый старт

Проверить генерацию карты в GUI:

```bash
python visualization/preview_scenarios.py --stage pretrain --obstacle-type cylinders --duration 5
```

Обучить базовую модель на пустой арене:

```bash
python training/train.py --stage 0 --eval
```

Обучить `stage 1` с RRT* и transfer learning из `stage 0`, если checkpoint найден:

```bash
python training/train.py --stage 1 --eval
```

Обучить pretrain на цилиндрах без планировщика:

```bash
python training/train.py --stage pretrain --obstacle-type cylinders --timesteps 1000000 --eval
```

Запустить RecurrentPPO для динамических карт:

```bash
python training/train.py \
  --algo recurrent_ppo \
  --stage pretrain \
  --obstacle-type dynamic_mix \
  --timesteps 1000000 \
  --eval
```

Открыть TensorBoard:

```bash
tensorboard --logdir logs/
```

## Обучение

Основная точка входа:

```bash
python training/train.py --stage {0,1,pretrain} [options]
```

Частые параметры:

| Параметр | Что делает |
| --- | --- |
| `--algo ppo|recurrent_ppo` | Выбор PPO или LSTM-версии |
| `--obstacle-type TYPE` | Тип карты для `pretrain` |
| `--no-planner` | Отключить RRT* в `stage 0/1` |
| `--timesteps N` | Бюджет шагов, при `--continue` это дополнительный бюджет |
| `--n-envs N` | Число параллельных сред |
| `--continue` | Продолжить обучение из checkpoint |
| `--init-model PATH` | Явно указать модель для старта/продолжения |
| `--init-normalize PATH` | Явно указать `VecNormalize` stats |
| `--watch` | GUI-режим с фиксированной картой и траекторией |
| `--fixed-map` | Не пересоздавать карту между эпизодами |
| `--swarm-drones N` | Несколько дронов в одной общей карте |
| `--eval` | Включить периодическую оценку и best checkpoint |
| `--promote-best-to-main` | После обучения скопировать лучший eval checkpoint в основной путь |
| `--artifact-tag TAG` | Добавить suffix к checkpoint-файлам |
| `--run-tag TAG` | Добавить suffix к логам TensorBoard/eval/diagnostics |
| `--no-diag` | Отключить структурные diagnostics logs |
| `--safety-shield` | Включить action filter от опасных сближений |

По умолчанию диагностика включена, а периодическая оценка выключена. Для воспроизводимых экспериментов лучше запускать с `--eval`, `--run-tag` и `--artifact-tag`.

Пример длинного запуска с лучшим checkpoint как основным артефактом:

```bash
python training/train.py \
  --algo recurrent_ppo \
  --stage pretrain \
  --obstacle-type beams \
  --timesteps 2000000 \
  --n-envs 8 \
  --eval \
  --eval-freq 50000 \
  --eval-episodes 20 \
  --promote-best-to-main \
  --artifact-tag beams_v1 \
  --run-tag beams_v1
```

GUI watch-режим:

```bash
python training/train.py --stage pretrain --obstacle-type cylinders --watch --watch-fps 60
```

Swarm на одной карте:

```bash
python training/train.py \
  --stage pretrain \
  --obstacle-type cylinders \
  --watch \
  --swarm-drones 8 \
  --watch-fps 90 \
  --no-show-paths
```

## Curriculum

`training/curriculum_train.py` запускает последовательное обучение по типам препятствий: обучает один тип, оценивает success rate, затем переходит к следующему.

Пример ночного запуска с автогенерацией графиков:

```bash
python training/curriculum_train.py \
  --algo recurrent_ppo \
  --allow-scratch \
  --chunk-timesteps 1000000 \
  --max-rounds-per-stage 3 \
  --n-envs 8 \
  --make-plots \
  --run-tag curriculum_v1 \
  --artifact-tag curriculum_v1
```

Если есть стартовая модель:

```bash
python training/curriculum_train.py \
  --algo recurrent_ppo \
  --start-model models/recurrent_ppo_pretrain_cylinders_enhanced_obs.zip \
  --start-normalize models/vec_normalize_recurrent_ppo_pretrain_cylinders_enhanced_obs.pkl
```

## Оценка моделей

Headless-оценка без GUI:

```bash
python training/evaluate.py \
  --model models/ppo_pretrain_cylinders_enhanced_obs.zip \
  --normalize models/vec_normalize_pretrain_cylinders_enhanced_obs.pkl \
  --stage pretrain \
  --obstacle-type cylinders \
  --episodes 100 \
  --json-out reports/eval_cylinders_100.json
```

Для RecurrentPPO:

```bash
python training/evaluate.py \
  --algo recurrent_ppo \
  --model models/recurrent_ppo_pretrain_dynamic_mix_enhanced_obs.zip \
  --normalize models/vec_normalize_recurrent_ppo_pretrain_dynamic_mix_enhanced_obs.pkl \
  --stage pretrain \
  --obstacle-type dynamic_mix \
  --episodes 100
```

Для `stage 0/1` с RRT*:

```bash
python training/evaluate.py \
  --model models/ppo_drone_nav_stage1_planner.zip \
  --normalize models/vec_normalize_stage1_planner.pkl \
  --stage 1 \
  --planner \
  --episodes 50
```

Важно: при запуске обученной модели почти всегда нужно загружать соответствующий `.pkl` файл `VecNormalize`. Без него observation будет в другом масштабе, и поведение модели может резко ухудшиться.

## Визуализация

Просмотр обученной pretrain-модели в PyBullet GUI:

```bash
python visualization/visualize.py \
  --model models/ppo_pretrain_cylinders_enhanced_obs.zip \
  --normalize models/vec_normalize_pretrain_cylinders_enhanced_obs.pkl \
  --stage pretrain \
  --obstacle-type cylinders \
  --episodes 3 \
  --watch-fps 60
```

Просмотр `stage 1` с RRT* и отрисовкой planned path:

```bash
python visualization/visualize_planner.py \
  --model models/ppo_drone_nav_stage1_planner.zip \
  --normalize models/vec_normalize_stage1_planner.pkl \
  --stage 1 \
  --n_episodes 3
```

Предпросмотр генерации сценариев без модели:

```bash
python visualization/preview_scenarios.py --stage pretrain --obstacle-type beams --continuous
```

Рендер карт для отчета:

```bash
python visualization/render_env_screenshots.py \
  --maps cylinders beams swinging_sticks city_dynamic \
  --output-dir reports/presentation_env_screenshots
```

## Графики и отчеты

Генерация графиков по TensorBoard, eval logs и structured diagnostics:

```bash
python training/generate_training_plots.py \
  --logs-dir logs \
  --eval-dir logs/eval \
  --diagnostics-dir logs/training_diagnostics \
  --reports-dir reports/training_plots \
  --tag current_run \
  --formats png,svg
```

Презентационный стиль:

```bash
python training/generate_training_plots.py --presentation --tag vkr_figures
```

Сравнительные графики по выбранным картам:

```bash
python training/plot_selected_map_comparison.py
```

## Где лежат результаты

Основные checkpoint-пути:

```text
models/
├── ppo_drone_nav_stage0_planner.zip
├── vec_normalize_stage0_planner.pkl
├── ppo_drone_nav_stage1_planner.zip
├── vec_normalize_stage1_planner.pkl
├── ppo_pretrain_<obstacle>_enhanced_obs.zip
├── vec_normalize_pretrain_<obstacle>_enhanced_obs.pkl
├── recurrent_ppo_pretrain_<obstacle>_enhanced_obs.zip
└── vec_normalize_recurrent_ppo_pretrain_<obstacle>_enhanced_obs.pkl
```

При `--eval` дополнительно создаются:

```text
models/best_checkpoints/<run_name>/best_model.zip
models/best_checkpoints/<run_name>/best_model_vecnormalize.pkl
models/last_checkpoints/<run_name>/last_model.zip
models/last_checkpoints/<run_name>/last_model_vecnormalize.pkl
logs/eval/<run_name>/eval_history.jsonl
```

Structured diagnostics:

```text
logs/training_diagnostics/<run_name>_<timestamp>/
├── run_config.json
├── episodes_compact.jsonl
├── milestones.jsonl
├── bad_episodes_top.jsonl
└── summary_blocks.json
```

`episodes_compact.jsonl` хранит все неуспешные эпизоды и сэмпл успешных, `bad_episodes_top.jsonl` ранжирует тяжелые провалы, а `summary_blocks.json` агрегирует причины и reward-компоненты.

## Тесты

Запустить все тесты:

```bash
python -m pytest tests
```

Только runtime/reward/observation тесты:

```bash
python -m pytest tests/test_nav_runtime_observation_reward.py
```

RRT* тесты:

```bash
python -m pytest tests/test_rrt_star.py
```

Часть RRT* тестов строит графики через matplotlib, поэтому при запуске на сервере может понадобиться headless backend (`MPLBACKEND=Agg`) или отключенный GUI.

## Конфигурация

Главные файлы:

- `config/base.py` - частоты симуляции, лимиты скоростей, PPO-параметры, reward scales, safety shield;
- `config/stage0.py` - пустая арена и RRT* параметры;
- `config/stage1.py` - статические цилиндры и RRT* параметры;
- `config/pretrain.py` - коридор, enhanced observations, типы карт и динамические препятствия;
- `config/debug.py` - включение подробных reward/action/navigation metrics;
- `config/runtime_sync.py` - синхронизация stage-конфига в worker-процессах `SubprocVecEnv`.

При изменении observation shape, например `USE_ENHANCED_OBS` или числа raycast-лучей, старые модели становятся несовместимыми с новой средой. Для таких экспериментов используйте новый `--artifact-tag`.

## Важные замечания

- `pretrain` всегда работает без RRT*. Планировщик предназначен для `stage 0/1`.
- `NavAviaryWithPlanner` использует RRT* как глобальный waypoint planner; policy учится следовать текущему waypoint.
- RRT* collision checking реализован через цилиндрическую аппроксимацию препятствий, поэтому он подходит прежде всего для `stage 1`.
- `--watch` принудительно использует одну среду (`DummyVecEnv`) и фиксированную карту.
- `--swarm-drones` работает только без планировщика и хранит параллелизм внутри одной среды.
- Safety shield по умолчанию выключен для обучения с нуля, но его можно включать на оценке/визуализации.
- `COMMANDS.md` и `PARAMETERS_GUIDE.md` полезны как исторические заметки, но точные текущие значения лучше проверять в `config/` и `training/train.py`.

## Рекомендуемые рабочие сценарии

Минимальная проверка среды:

```bash
python -m pytest tests/test_nav_runtime_observation_reward.py
python visualization/preview_scenarios.py --stage pretrain --obstacle-type empty --duration 3
```

Классический pipeline stage 0 -> stage 1:

```bash
python training/train.py --stage 0 --eval --promote-best-to-main
python training/train.py --stage 1 --eval --promote-best-to-main
python training/evaluate.py --model models/ppo_drone_nav_stage1_planner.zip --normalize models/vec_normalize_stage1_planner.pkl --stage 1 --planner --episodes 100
```

Pipeline для pretrain-карты:

```bash
python training/train.py --stage pretrain --obstacle-type cylinders --eval --promote-best-to-main
python training/evaluate.py --model models/ppo_pretrain_cylinders_enhanced_obs.zip --normalize models/vec_normalize_pretrain_cylinders_enhanced_obs.pkl --stage pretrain --obstacle-type cylinders --episodes 100
python visualization/visualize.py --model models/ppo_pretrain_cylinders_enhanced_obs.zip --normalize models/vec_normalize_pretrain_cylinders_enhanced_obs.pkl --stage pretrain --obstacle-type cylinders
```
