#!/usr/bin/env python
# Test script for model evaluation pipeline

from src.evaluation import evaluate_best_model

print('🔍 Testing model evaluation pipeline...')
try:
    metrics = evaluate_best_model()
    print('✅ Model evaluation completed successfully!')
    print('\n📊 Key Metrics:')
    print(f'  R² Score: {metrics["r2"]:.4f}')
    print(f'  RMSE: ${metrics["rmse"]:,.2f}')
    print(f'  MAE: ${metrics["mae"]:,.2f}')
    print(f'  MAPE: {metrics["mape"]:.2f}%')
    print(f'\n📁 Reports and plots saved to docs/ directory')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
