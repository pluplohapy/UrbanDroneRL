# Quick Reference - Команды для обучения

**🆕 Все stages теперь используют RRT* по умолчанию!**

## 🚀 Быстрый старт

### 1. Тест RRT* (5 минут)
```bash
python quick_test_rrt.py
```

### 2. Обучение Stage 0 с RRT* (30-60 минут)
```bash
python training/train.py --stage 0
```

### 3. Обучение Stage 1 с RRT* (2-3 часа)
```bash
python training/train.py --stage 1
```

### 4. Мониторинг (в другом терминале)
```bash
tensorboard --logdir logs/
# http://localhost:6006
```

## 📊 Все команды обучения

```bash
# Stage 0: С RRT* планировщиком (рекомендуется)
python training/train.py --stage 0

# Stage 0: БЕЗ планировщика (baseline)
python training/train.py --stage 0 --no-planner

# Stage 1: С RRT* планировщиком (рекомендуется)
python training/train.py --stage 1

# Stage 1: БЕЗ планировщика (для сравнения)
python training/train.py --stage 1 --no-planner

# Кастомное количество шагов
python training/train.py --stage 1 --timesteps 2000000

# Больше параллельных сред (быстрее, но больше RAM)
python training/train.py --stage 1 --n-envs 8

# Продолжить прерванное обучение
python training/train.py --stage 1 --continue
```

## 👁️ Визуализация

```bash
# Stage 0 с планировщиком
python visualize_planner.py --model models/ppo_drone_nav_stage0_planner --episodes 5

# Stage 0 без планировщика
python visualize.py --model models/ppo_drone_nav_stage0 --episodes 5

# Stage 1 с планировщиком
python visualize_planner.py --model models/ppo_drone_nav_stage1_planner --episodes 5

# Stage 1 без планировщика
python visualize.py --model models/ppo_drone_nav_stage1 --episodes 5
```

## 🧪 Тестирование

```bash
# Быстрый тест RRT* (5 минут)
python quick_test_rrt.py

# Полный тест RRT* (10 минут)
python test_rrt_star.py

# Проверка 5м лимита waypoints
python check_waypoint_distances.py

# Сравнение с/без планировщика
python compare_planner.py
```

## 📈 Мониторинг прогресса

### В консоли
Каждые 10 эпизодов выводится:
```
Episode   10 | Steps:   20480
  Outcomes : S= 20% | C= 30% | T= 50%
  Planner  : failures=  5% | avg_waypoints=4.2
  Reward: 156.30 | Length: 245.0
```

### TensorBoard
```bash
tensorboard --logdir logs/
# Открыть http://localhost:6006
```

### Файлы моделей
```
models/
├── ppo_drone_nav_stage0.zip              # Stage 0
├── ppo_drone_nav_stage1_planner.zip      # Stage 1 с RRT*
└── ppo_drone_nav_stage1.zip              # Stage 1 без RRT*
```

## 🎯 Рекомендуемый порядок

```bash
# 1. Тест планировщика
python quick_test_rrt.py

# 2. Обучить Stage 0 с RRT* (базовые навыки + waypoint following)
python training/train.py --stage 0

# 3. Визуализировать Stage 0
python visualize_planner.py --model models/ppo_drone_nav_stage0_planner

# 4. Обучить Stage 1 с transfer learning
python training/train.py --stage 1

# 5. Мониторить в TensorBoard
tensorboard --logdir logs/

# 6. Визуализировать Stage 1
python visualize_planner.py --model models/ppo_drone_nav_stage1_planner

# 7. Сравнить подходы
python compare_planner.py
```

## ⚙️ Настройки

### Ускорить планирование Stage 0
Отредактировать `training/train.py` строка 215:
```python
'max_iter': 300,  # вместо 500 для Stage 0
```

### Ускорить планирование Stage 1
Отредактировать `training/train.py` строка 254:
```python
'max_iter': 500,  # вместо 1000
```

### Изменить параметры обучения
Отредактировать `config.py`:
```python
N_ENVS = 8              # Больше сред = быстрее
TOTAL_STEPS = 1000000   # Больше шагов = лучше качество
```

### Отключить debug логи
В `config.py`:
```python
DEBUG_MODE = False
```

## 🐛 Troubleshooting

### Планирование слишком долгое
```bash
# Уменьшить max_iter в training/train.py
'max_iter': 500  # вместо 1000
```

### Out of memory
```bash
# Уменьшить количество сред
python training/train.py --stage 1 --n-envs 2
```

### Обучение прервалось
```bash
# Продолжить с checkpoint
python training/train.py --stage 1 --continue
```

### Низкий success rate
- Проверить TensorBoard графики
- Убедиться что Stage 0 обучен
- Увеличить timesteps: `--timesteps 2000000`

## 📚 Документация

- `training/README.md` - Подробная документация по обучению
- `RRT_STAR_README.md` - Документация RRT* планировщика
- `USAGE.md` - Общее руководство
- `README.md` - Обзор проекта

## 💡 Полезные советы

1. **Всегда обучайте Stage 0 первым** - это база для Stage 1
2. **Используйте TensorBoard** - видите прогресс в реальном времени
3. **Сохраняйте checkpoints** - обучение можно прервать
4. **Сравнивайте подходы** - понимайте влияние RRT*
5. **Проверяйте визуализацию** - убедитесь что агент работает правильно

## 🎓 Для диплома

```bash
# 1. Обучить обе версии
python training/train.py --stage 0
python training/train.py --stage 1
python training/train.py --stage 1 --no-planner

# 2. Сравнить результаты
python compare_planner.py

# 3. Собрать метрики из TensorBoard
tensorboard --logdir logs/

# 4. Создать визуализации
python visualize_planner.py --model models/ppo_drone_nav_stage1_planner --episodes 10
```

Результаты:
- Графики обучения (TensorBoard)
- Success rate: с RRT* vs без RRT*
- Траектории полетов
- Время планирования
- Метрики навигации
