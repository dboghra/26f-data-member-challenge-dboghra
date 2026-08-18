import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from clean_data import load_and_clean_data
from linear_regression import train_linear_regression, evaluate_model as eval_lr, compare_to_baseline as baseline_lr
from random_forest import train_random_forest, evaluate_model as eval_rf, compare_to_baseline as baseline_rf


# Reusable accuracy score function
def calculate_accuracy_score(y_test, y_pred):
    """Calculate metrics - REUSABLE across models"""
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'r2': r2, 'mae': mae, 'rmse': rmse}

def main():
    print("="*70)
    print("DEVELOPER SALARY PREDICTION - FULL PIPELINE")
    print("="*70)
    print()
    
    # Load data
    print("STEP 1: Loading and cleaning data...\n")
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data('data/survey.csv')
    
    print("\n")
    
    # Linear Regression
    print("STEP 2: Training Linear Regression...\n")
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = eval_lr(lr_model, X_test, y_test)
    baseline_lr(y_test, lr_metrics['predictions'])
    
    print("\n")
    
    # Random Forest
    print("STEP 3: Training Random Forest...\n")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = eval_rf(rf_model, X_test, y_test)
    baseline_rf(y_test, rf_metrics['predictions'])
    
    print("\n")
    
    # Compare metrics (reusable function)
    print("STEP 4: Comparing models...\n")
    lr_acc = calculate_accuracy_score(y_test, lr_metrics['predictions'])
    rf_acc = calculate_accuracy_score(y_test, rf_metrics['predictions'])
    
    # Print comparison
    print("="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print()
    print(f"{'Metric':<25} {'Linear Regression':<25} {'Random Forest':<25}")
    print("-"*70)
    print(f"{'R² Score':<25} {lr_acc['r2']:<25.4f} {rf_acc['r2']:<25.4f}")
    print(f"{'MAE':<25} ${lr_acc['mae']:<24,.0f} ${rf_acc['mae']:<24,.0f}")
    print(f"{'RMSE':<25} ${lr_acc['rmse']:<24,.0f} ${rf_acc['rmse']:<24,.0f}")
    print()
    
    # Determine winner
    if rf_acc['mae'] < lr_acc['mae']:
        difference = lr_acc['mae'] - rf_acc['mae']
        print(f"🏆 Random Forest wins! (${difference:,.0f} lower MAE)")
    elif lr_acc['mae'] < rf_acc['mae']:
        difference = rf_acc['mae'] - lr_acc['mae']
        print(f"🏆 Linear Regression wins! (${difference:,.0f} lower MAE)")
    else:
        print("🤝 Tie!")

if __name__ == '__main__':
    main()
    