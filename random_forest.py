import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a random forest model on training data.
    
    Random Forest is an ensemble method that:
    - Builds multiple decision trees
    - Each tree learns from a random subset of features and data
    - Final prediction is the average of all tree predictions
    - More robust to overfitting than single decision trees
    
    Args:
        X_train (pd.DataFrame): Training features (n_samples × n_features)
        y_train (pd.Series): Training target values (salary)
        n_estimators (int): Number of trees to build (default 100)
        random_state (int): Seed for reproducibility (default 42)
        
    Returns:
        RandomForestRegressor: Fitted model ready for predictions
    """
    print("="*70)
    print("RANDOM FOREST MODEL")
    print("="*70)
    print()
    
    print(f"Training random forest with {n_estimators} trees...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores for faster training
        verbose=0   # Don't print tree-by-tree progress
    )
    model.fit(X_train, y_train)
    print(f"✓ Model trained successfully ({n_estimators} trees)\n")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model (RandomForestRegressor): Fitted model
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
    Extract feature importance from the trained random forest model.
    
    In random forest, feature importance is based on how much each feature
    decreases impurity (variance) across all trees in the forest.
    
    Unlike linear regression coefficients, these importances are:
    - Always non-negative (0 to 1, sum to 1.0)
    - Relative importance (percentage contribution)
    - Harder to interpret (not in dollars)
    
    Example:
        Remote_Remote importance = 0.25
        Means: This feature accounts for 25% of the variance explained
        
    Args:
        model (RandomForestRegressor): Fitted model
        feature_names (list): List of feature column names
        
    Returns:
        pd.DataFrame: DataFrame with features and their importance scores,
                     sorted by importance (largest first)
    """
    # Create dataframe with feature names and importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    
    # Sort by importance (largest first)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Reset index for clean display
    importance_df = importance_df.reset_index(drop=True)
    
    return importance_df


def print_feature_importance(model, feature_names):
    """
    Print feature importance in a readable format.
    
    Shows which features have the biggest impact on salary predictions.
    
    Args:
        model (RandomForestRegressor): Fitted model
        feature_names (list): List of feature column names
    """
    importance_df = get_feature_importance(model, feature_names)
    
    print("Feature Importance (Random Forest)")
    print("="*70)
    print("\nHow much each feature contributes to predictions:")
    print("(Higher = more important. All sum to 1.0)")
    print("-"*70)
    
    for idx, row in importance_df.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        percentage = importance * 100
        
        # Create a simple bar chart
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length
        
        # Format the output nicely
        print(f"{feature:<50} {percentage:>6.2f}%  {bar}")
    
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
    print(f"Random Forest Model:                    ${model_mae:,.0f} MAE")
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
    Test the random forest module standalone.
    
    Run this to test:
    $ python random_forest.py
    """
    from clean_data import load_and_clean_data
    
    # Load and clean data
    print("Loading and cleaning data...\n")
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data('survey.csv')
    
    print("\n")
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Show feature importance
    print_feature_importance(model, feature_names)
    
    # Compare to baseline
    compare_to_baseline(y_test, metrics['predictions'])
    
    # Show example predictions
    print_prediction_examples(y_test, metrics['predictions'], n_examples=15)

