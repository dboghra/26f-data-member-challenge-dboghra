# random forest logic here 
def train_random_forest(X_train, y_train, n_estimators=100):
    # Train model
    ...

def evaluate_model(model, X_test, y_test):
    # Get metrics, predictions
    ...

def compare_to_baseline(y_test, y_pred):
    # Compare this model to baseline
    ...

if __name__ == '__main__':
    # Test standalone
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data('survey.csv')
    model = train_random_forest(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    compare_to_baseline(y_test, metrics['predictions'])  # ← Called here