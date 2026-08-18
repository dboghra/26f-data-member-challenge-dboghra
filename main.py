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
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data('survey.csv')
    
    # Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = eval_lr(lr_model, X_test, y_test)
    baseline_lr(y_test, lr_metrics['predictions'])  # ← Model handles its own baseline
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = eval_rf(rf_model, X_test, y_test)
    baseline_rf(y_test, rf_metrics['predictions'])  # ← Model handles its own baseline
    
    # Compare metrics (reusable function)
    lr_acc = calculate_accuracy_score(y_test, lr_metrics['predictions'])
    rf_acc = calculate_accuracy_score(y_test, rf_metrics['predictions'])
    
    # Print comparison
    print("\nFinal Comparison:")
    print(f"Linear Regression MAE: ${lr_acc['mae']:,.0f}")
    print(f"Random Forest MAE: ${rf_acc['mae']:,.0f}")

if __name__ == '__main__':
    main()