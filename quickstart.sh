#!/bin/bash
# Quick start script for training drone navigation

echo "============================================================"
echo "DRONE NAVIGATION - QUICK START"
echo "============================================================"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "drone_nav" ]]; then
    echo "⚠️  Warning: conda environment 'drone_nav' is not activated"
    echo "   Run: conda activate drone_nav"
    echo ""
    exit 1
fi

echo "Select training mode:"
echo "  1) Test RRT* planner (5 minutes)"
echo "  2) Train Stage 0 - Empty arena (30-60 min)"
echo "  3) Train Stage 1 - With RRT* planner (2-3 hours)"
echo "  4) Train Stage 1 - Without planner (2-3 hours)"
echo "  5) Visualize Stage 0"
echo "  6) Visualize Stage 1 with planner"
echo "  7) Compare approaches"
echo "  8) Start TensorBoard"
echo ""
read -p "Enter choice [1-8]: " choice

case $choice in
    1)
        echo ""
        echo "Running RRT* quick test..."
        python quick_test_rrt.py
        ;;
    2)
        echo ""
        echo "Training Stage 0 (empty arena)..."
        echo "Expected time: 30-60 minutes"
        echo "Press Ctrl+C to stop"
        echo ""
        python training/train.py --stage 0
        ;;
    3)
        echo ""
        echo "Training Stage 1 with RRT* planner..."
        echo "Expected time: 2-3 hours"
        echo "Press Ctrl+C to stop"
        echo ""
        python training/train.py --stage 1
        ;;
    4)
        echo ""
        echo "Training Stage 1 without planner..."
        echo "Expected time: 2-3 hours"
        echo "Press Ctrl+C to stop"
        echo ""
        python training/train.py --stage 1 --no-planner
        ;;
    5)
        echo ""
        if [ -f "models/ppo_drone_nav_stage0.zip" ]; then
            echo "Visualizing Stage 0..."
            python visualize.py --model models/ppo_drone_nav_stage0 --episodes 5
        else
            echo "❌ Stage 0 model not found. Train it first:"
            echo "   python training/train.py --stage 0"
        fi
        ;;
    6)
        echo ""
        if [ -f "models/ppo_drone_nav_stage1_planner.zip" ]; then
            echo "Visualizing Stage 1 with planner..."
            python visualize_planner.py --model models/ppo_drone_nav_stage1_planner --episodes 5
        else
            echo "❌ Stage 1 planner model not found. Train it first:"
            echo "   python training/train.py --stage 1"
        fi
        ;;
    7)
        echo ""
        echo "Comparing approaches..."
        python compare_planner.py
        ;;
    8)
        echo ""
        echo "Starting TensorBoard..."
        echo "Open browser: http://localhost:6006"
        echo "Press Ctrl+C to stop"
        echo ""
        tensorboard --logdir logs/
        ;;
    *)
        echo ""
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"
