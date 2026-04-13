#!/bin/bash
# Скрипт для запуска полного обучения 600k шагов

echo "=========================================="
echo "Запуск полного обучения (600k шагов)"
echo "Ожидаемое время: ~3-4 часа"
echo "=========================================="

# Активировать окружение
source ~/miniforge3/etc/profile.d/conda.sh
conda activate drone_nav

# Проверить, что баг исправлен
echo ""
echo "Проверка исправления бага..."
python -c "
from envs.nav_aviary import NavAviary
from scenarios.stage0_empty import Stage0Scenario
env = NavAviary(Stage0Scenario(seed=0), gui=False)
assert hasattr(env, 'control_step_counter'), 'Баг не исправлен!'
print('✓ Баг исправлен: control_step_counter найден')
env.close()
"

# Обновить config для 600k шагов
echo ""
echo "Обновление конфигурации..."
python -c "
import config
print(f'Текущие настройки:')
print(f'  TOTAL_STEPS: {config.TOTAL_STEPS:,}')
print(f'  MAX_STEPS: {config.MAX_STEPS}')
print(f'  N_ENVS: {config.N_ENVS}')
"

# Запустить обучение
echo ""
echo "Запуск обучения..."
echo "Для мониторинга откройте в другом терминале:"
echo "  tensorboard --logdir=./logs/"
echo ""

python train.py

echo ""
echo "=========================================="
echo "Обучение завершено!"
echo "=========================================="
echo ""
echo "Для визуализации:"
echo "  python visualize.py"
echo ""
echo "Для оценки:"
echo "  python evaluate.py"
