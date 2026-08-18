#linear regression  logic here

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model on training data.
    Args:
        X_train (pd.DataFrame): Training features (n_samples × n_features)
        y_train (pd.Series): Training target values (salary)  
    Returns:LinearRegression: Fitted model ready for predictions
    """
    print("="*70)
    print("LINEAR REGRESSION MODEL")
    print("="*70)
    print()
    
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✓ Model trained successfully\n")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model (LinearRegression): Fitted model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values (actual salaries)
        
    Returns:
        dict: Dictionary containing:
            - 'r2': R² score (0-1, higher is better)
            - 'mae': Mean Absolute Error (in dollars)
            - 'rmse': Root Mean Squared Error (in dollars)
            - 'predictions': Predicted salaries for test set
            - 'residuals': Actual - Predicted (errors)
    """
    print(f"Evaluating on test set ({len(X_test)} samples)...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    residuals = y_test - y_pred
    
    # Print results
    print()
    print("  R² Score:  {:.4f}  ({:.1f}% of variance explained)".format(r2, r2 * 100))
    print(f"  MAE:       ${mae:,.0f}  (average prediction error)")
    print(f"  RMSE:      ${rmse:,.0f}  (penalizes large errors more)")
    print()
    
    # Store metrics in dictionary
    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred,
        'residuals': residuals,
        'y_test': y_test,
        'model': model
    }
    
    return metrics


def get_feature_importance(model, feature_names):
    """
    Extract feature importance (coefficients) from the trained model.
    Args:
        model (LinearRegression): Fitted model
        feature_names (list): List of feature column names
    Returns:
        pd.DataFrame: DataFrame with features and their coefficients,
                     sorted by absolute value (largest impact first)
    """
    # Create dataframe with feature names and coefficients
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    
    # Sort by absolute value of coefficient (largest impact first)
    importance_df = importance_df.sort_values('Coefficient', key=abs, ascending=False)
    
    # Reset index for clean display
    importance_df = importance_df.reset_index(drop=True)
    
    return importance_df


def print_feature_importance(model, feature_names):
    """
    Print feature importance in a readable format.
    
    Shows which features have the biggest impact on salary predictions.
    
    Args:
        model (LinearRegression): Fitted model
        feature_names (list): List of feature column names
    """
    importance_df = get_feature_importance(model, feature_names)
    
    print("Feature Importance (Model Coefficients)")
    print("="*70)
    print(f"\nBase salary (intercept): ${model.intercept_:,.0f}")
    print("(This is the predicted salary when all features are 0)\n")
    
    print("How each feature affects salary prediction (in dollars):")
    print("-"*70)
    
    for idx, row in importance_df.iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        direction = "↑" if coef > 0 else "↓"
        
        # Format the output nicely
        print(f"{direction} {feature:<50} ${coef:>12,.0f}")
    
    print()


def compare_to_baseline(y_test, y_pred):
    """
    Compare model performance to a naive baseline.
    Baseline = always predicting the mean salary
    Args:
        y_test (pd.Series): Actual test salaries
        y_pred (np.array): Model predictions 
    Returns:
        dict: Comparison metrics
    """
    # Baseline: always predict mean
    baseline_pred = np.full_like(y_test, y_test.mean())
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    
    # Model performance
    model_mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate improvement
    improvement_pct = ((baseline_mae - model_mae) / baseline_mae) * 100
    
    print("Comparison to Baseline")
    print("="*70)
    print(f"\nBaseline (always predict mean salary): ${baseline_mae:,.0f} MAE")
    print(f"Linear Regression Model:                ${model_mae:,.0f} MAE")
    print(f"Improvement:                            {improvement_pct:+.1f}%")
    print()
    
    return {
        'baseline_mae': baseline_mae,
        'model_mae': model_mae,
        'improvement_pct': improvement_pct
    }


def print_prediction_examples(y_test, y_pred, n_examples=10):
    """
    Print example predictions to see how the model performs.
    
    Args:
        y_test (pd.Series): Actual salaries
        y_pred (np.array): Predicted salaries
        n_examples (int): Number of examples to show (default 10)
    """
    print("Sample Predictions on Test Set")
    print("="*70)
    print(f"\n{'Actual Salary':<18} {'Predicted':<18} {'Error':<18} {'Error %':<10}")
    print("-"*70)
    
    # Randomly select examples
    n_show = min(n_examples, len(y_test))
    indices = np.random.choice(len(y_test), n_show, replace=False)
    
    for i in indices:
        actual = y_test.iloc[i]
        pred = y_pred[i]
        error = pred - actual
        error_pct = (error / actual) * 100 if actual != 0 else 0
        
        print(f"${actual:>15,.0f} ${pred:>15,.0f} ${error:>15,.0f} {error_pct:>8.1f}%")
    
    print()


if __name__ == '__main__':
    """
    Test the linear regression module standalone.
    
    Run this to test:
    $ python linear_regression.py
    """
    from clean_data import load_and_clean_data
    
    # Load and clean data
    print("Loading and cleaning data...\n")
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data('survey.csv')
    
    print("\n")
    
    # Train model
    model = train_linear_regression(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Show feature importance
    print_feature_importance(model, feature_names)
    
    # Compare to baseline
    compare_to_baseline(y_test, metrics['predictions'])
    
    # Show example predictions
    print_prediction_examples(y_test, metrics['predictions'], n_examples=15)