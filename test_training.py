#!/usr/bin/env python
# Test script for model training pipeline

from src.model_training import train_all_models

print('Testing model training pipeline...')
try:
    metrics = train_all_models()
    print('✅ Model training completed successfully!')
    print('Model metrics:')
    for model, metric in metrics.items():
        print(f'  {model}: R² = {metric["r2"]:.4f}, RMSE = {metric["rmse"]:.2f}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
