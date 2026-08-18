import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load the survey data from CSV.
    Args:filepath (str): Path to the CSV file   
    Returns:pd.DataFrame: Raw data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns\n")
    return df


def remove_missing_salary(df):
    """
    Remove rows where the target variable (salary) is missing.
    Args:df (pd.DataFrame): Raw data
        
    Returns:pd.DataFrame: Data with valid salary values
    """
    initial_count = len(df)
    df = df.dropna(subset=['annual_salary_usd'])
    removed = initial_count - len(df)
    
    print(f"Removing rows with missing salary...")
    print(f"  Removed: {removed} rows")
    print(f"  Remaining: {len(df)} rows\n")
    
    return df


def remove_missing_experience(df):
    """
    Remove rows with missing work experience or coding experience.
    These are critical features that can't be reasonably estimated.
    Args:df (pd.DataFrame): Data with valid salaries  
    Returns:pd.DataFrame: Data with valid experience values
    """
    initial_count = len(df)
    df = df.dropna(subset=['WorkExp', 'YearsCode'])
    removed = initial_count - len(df)
    
    print(f"Removing rows with missing experience values...")
    print(f"  Removed: {removed} rows")
    print(f"  Remaining: {len(df)} rows\n")
    
    return df


def remove_missing_education(df):
    """
    Remove rows with missing education level.
    Education is a key predictor and should not be imputed.
    Args:df (pd.DataFrame): Data with valid experience
        
    Returns:pd.DataFrame: Data with valid education values
    """
    initial_count = len(df)
    df = df.dropna(subset=['EdLevel'])
    removed = initial_count - len(df)
    
    print(f"Removing rows with missing education level...")
    print(f"  Removed: {removed} rows")
    print(f"  Remaining: {len(df)} rows\n")
    
    return df


def encode_age(df):
    """
    Convert age ranges (categorical) to single numeric values (midpoints).
    Example:
        '18-24 years old' → 21
        '25-34 years old' → 29.5
        '35-44 years old' → 39.5
    Args:df (pd.DataFrame): Data with Age column 
    Returns:pd.DataFrame: Data with Age_numeric column added
    """
    age_mapping = {
        '18-24 years old': 21,
        '25-34 years old': 29.5,
        '35-44 years old': 39.5,
        '45-54 years old': 49.5,
        '55-64 years old': 59.5,
        '65 years or older': 70,
        'Prefer not to say': 40  # Use middle value for non-responses
    }
    
    df = df.copy()
    df['Age_numeric'] = df['Age'].map(age_mapping)
    
    # Fill any unmapped values with median
    if df['Age_numeric'].isnull().any():
        df['Age_numeric'] = df['Age_numeric'].fillna(df['Age_numeric'].median())
    
    print(f"Encoding age ranges to numeric values...")
    print(f"  Created: Age_numeric column")
    print(f"  Sample values: 21, 29.5, 39.5, 49.5, 59.5, 70\n")
    
    return df


def encode_education(df):
    """
    Convert education levels (categorical) to numeric ordinal scale (1-7).
    The scale respects the hierarchy of education:
        1 = Primary/elementary school
        2 = Secondary school (high school)
        3 = Some college
        4 = Associate degree
        5 = Bachelor's degree
        6 = Master's degree
        7 = Professional degree (PhD, MD, JD, etc.)
    Args:df (pd.DataFrame): Data with EdLevel column
    Returns:pd.DataFrame: Data with EdLevel_numeric column added
    """
    education_order = {
        'Primary/elementary school': 1,
        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 2,
        'Some college/university study without earning a degree': 3,
        'Associate degree (A.A., A.S., etc.)': 4,
        'Bachelor\'s degree (B.A., B.S., B.Eng., etc.)': 5,
        'Master\'s degree (M.A., M.S., M.Eng., MBA, etc.)': 6,
        'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 7,
        'Other (please specify):': 3  # Treat "other" as some college
    }
    
    df = df.copy()
    
    # Handle smart quotes in the data (common encoding issue)
    df['EdLevel_clean'] = df['EdLevel'].str.replace(''', "'").str.replace(''', "'")
    
    # Map to numeric scale
    df['EdLevel_numeric'] = df['EdLevel_clean'].map(education_order)
    
    # Fill any unmapped values with mode (most common education level)
    if df['EdLevel_numeric'].isnull().any():
        df['EdLevel_numeric'] = df['EdLevel_numeric'].fillna(df['EdLevel_numeric'].mode()[0])
    
    print(f"Encoding education levels to numeric scale (1-7)...")
    print(f"  Created: EdLevel_numeric column")
    print(f"  Scale: 1=Primary, 2=High School, ..., 7=Professional Degree\n")
    
    return df


def encode_employment(df):
    """
    Convert employment status to binary (0 or 1).
    1 = Employed (full-time)
    0 = Other (freelance, student, retired, etc.)
    Args:df (pd.DataFrame): Data with Employment column   
    Returns:pd.DataFrame: Data with IsEmployed column added
    """
    df = df.copy()
    df['IsEmployed'] = (df['Employment'] == 'Employed').astype(int)
    
    employed_count = df['IsEmployed'].sum()
    
    print(f"Encoding employment status to binary...")
    print(f"  Created: IsEmployed column (1=Employed, 0=Other)")
    print(f"  Employed: {employed_count}")
    print(f"  Other: {len(df) - employed_count}\n")
    
    return df


def encode_manager_status(df):
    """
    Convert role to binary manager status (0 or 1).
    1 = People manager
    0 = Individual contributor
    Args:df (pd.DataFrame): Data with ICorPM column  
    Returns:pd.DataFrame: Data with IsPeopleManager column added
    """
    df = df.copy()
    
    # Handle missing values: default to individual contributor
    df['ICorPM_filled'] = df['ICorPM'].fillna('Individual contributor')
    df['IsPeopleManager'] = (df['ICorPM_filled'] == 'People manager').astype(int)
    
    manager_count = df['IsPeopleManager'].sum()
    
    print(f"Encoding manager status to binary...")
    print(f"  Created: IsPeopleManager column (1=Manager, 0=Individual Contributor)")
    print(f"  Managers: {manager_count}")
    print(f"  Individual Contributors: {len(df) - manager_count}\n")
    
    return df


def encode_org_size(df):
    """
    Convert organization size to numeric ordinal scale (1-8).
    The scale reflects company size impact on salary:
        1 = Solo freelancer
        2 = Less than 20 employees
        3 = 20-99 employees
        4 = 100-499 employees
        5 = 500-999 employees
        6 = 1,000-4,999 employees
        7 = 5,000-9,999 employees
        8 = 10,000+ employees
    Args:df (pd.DataFrame): Data with OrgSize column
    Returns:pd.DataFrame: Data with OrgSize_numeric column added
    """
    org_order = {
        'Just me - I am a freelancer, sole proprietor, etc.': 1,
        'Less than 20 employees': 2,
        '20 to 99 employees': 3,
        '100 to 499 employees': 4,
        '500 to 999 employees': 5,
        '1,000 to 4,999 employees': 6,
        '5,000 to 9,999 employees': 7,
        '10,000 or more employees': 8,
        'I don\'t know': 4  # Use middle value for unknowns
    }
    
    df = df.copy()
    
    # Handle missing values: fill with mode (most common org size)
    org_mode = df['OrgSize'].mode()[0]
    df['OrgSize_filled'] = df['OrgSize'].fillna(org_mode)
    
    # Map to numeric scale
    df['OrgSize_numeric'] = df['OrgSize_filled'].map(org_order)
    
    # Fill any unmapped with default
    if df['OrgSize_numeric'].isnull().any():
        df['OrgSize_numeric'] = df['OrgSize_numeric'].fillna(4)
    
    print(f"Encoding organization size to numeric scale (1-8)...")
    print(f"  Created: OrgSize_numeric column")
    print(f"  Scale: 1=Solo, 2=<20, ..., 8=10,000+\n")
    
    return df


def encode_remote_work(df):
    """
    Convert remote work status to one-hot encoded binary columns.
    Creates separate binary columns for each remote work type:
    - Remote_In-person
    - Remote_Hybrid-remote
    - Remote_Hybrid-flexibility
    - Remote_YourChoice
    The "Remote" category is the reference (dropped to avoid collinearity).
    Args:df (pd.DataFrame): Data with RemoteWork column
    Returns:tuple: (df with original columns, dummy_df with binary columns)
    """
    df = df.copy()
    
    # Handle missing values: fill with mode (most common remote work type)
    remote_mode = df['RemoteWork'].mode()[0]
    df['RemoteWork_filled'] = df['RemoteWork'].fillna(remote_mode)
    
    # Create one-hot encoded columns (drop first to avoid collinearity)
    remote_dummies = pd.get_dummies(df['RemoteWork_filled'], prefix='Remote', drop_first=True)
    
    print(f"Encoding remote work status to binary columns...")
    print(f"  Created: {remote_dummies.shape[1]} binary columns")
    print(f"  Columns: {', '.join(remote_dummies.columns.tolist())}\n")
    
    return df, remote_dummies


def build_feature_matrix(df, remote_dummies):
    """
    Combine all encoded features into a single feature matrix.
    This includes:
    - Age_numeric
    - EdLevel_numeric
    - WorkExp (already numeric)
    - YearsCode (already numeric)
    - IsEmployed
    - IsPeopleManager
    - OrgSize_numeric
    - Remote work binary columns
    Args:
        df (pd.DataFrame): Data with encoded features
        remote_dummies (pd.DataFrame): One-hot encoded remote work columns  
    Returns:tuple: (X, y) where X is feature matrix and y is target (salary)
    """
    # Select numeric features
    X = pd.concat([
        df[['Age_numeric', 'EdLevel_numeric', 'WorkExp', 'YearsCode',
            'IsEmployed', 'IsPeopleManager', 'OrgSize_numeric']],
        remote_dummies
    ], axis=1)
    
    y = df['annual_salary_usd']
    
    print(f"Building feature matrix...")
    print(f"  X shape: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  y shape: {y.shape[0]} samples")
    print(f"\nFeature list:")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")
    print()
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (salary)
        test_size (float): Proportion of data to use for testing (default 0.2 = 20%)
        random_state (int): Seed for reproducibility (default 42)
    Returns:tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Splitting data into train/test sets (80/20)...")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Random state: {random_state} (for reproducibility)\n")
    
    return X_train, X_test, y_train, y_test


def load_and_clean_data(filepath, test_size=0.2, random_state=42):
    """
    Main function: Load raw data and perform all cleaning/encoding steps.
    This is the function you'll call from main.py.
    Args:
        filepath (str): Path to the survey CSV file
        test_size (float): Proportion for test set (default 0.2)
        random_state (int): Seed for reproducibility
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
               where feature_names is a list of feature column names
    """
    print("="*70)
    print("DATA CLEANING PIPELINE")
    print("="*70)
    print()
    
    # Load raw data
    df = load_data(filepath)
    
    # Remove incomplete records
    df = remove_missing_salary(df)
    df = remove_missing_experience(df)
    df = remove_missing_education(df)
    
    # Encode categorical variables
    df = encode_age(df)
    df = encode_education(df)
    df = encode_employment(df)
    df = encode_manager_status(df)
    df = encode_org_size(df)
    df, remote_dummies = encode_remote_work(df)
    
    # Build feature matrix
    X, y = build_feature_matrix(df, remote_dummies)
    
    # Verify no missing values
    if X.isnull().sum().sum() > 0:
        print("⚠️  WARNING: NaN values found in feature matrix!")
        print(X.isnull().sum()[X.isnull().sum() > 0])
    else:
        print("✓ No missing values in feature matrix\n")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    
    # Summary statistics
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Final dataset: {len(df)} samples")
    print(f"Features: {X.shape[1]}")
    print(f"Target variable: annual_salary_usd")
    print(f"\nTarget statistics:")
    print(f"  Mean: ${y.mean():,.0f}")
    print(f"  Median: ${y.median():,.0f}")
    print(f"  Min: ${y.min():,.0f}")
    print(f"  Max: ${y.max():,.0f}")
    print(f"  Std Dev: ${y.std():,.0f}")
    print()
    
    return X_train, X_test, y_train, y_test, list(X.columns)


def print_data_info(X_train, X_test, y_train, y_test):
    """
    Print detailed information about the split data.
    
    Useful for debugging and understanding your data.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target variables
    """
    print("="*70)
    print("DATA INFO")
    print("="*70)
    print(f"\nTraining Set:")
    print(f"  X_train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  y_train: {len(y_train)} samples")
    print(f"  Salary range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    print(f"  Mean salary: ${y_train.mean():,.0f}")
    
    print(f"\nTest Set:")
    print(f"  X_test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print(f"  y_test: {len(y_test)} samples")
    print(f"  Salary range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    print(f"  Mean salary: ${y_test.mean():,.0f}")
    print()


if __name__ == '__main__':
    """
    Test the cleaning pipeline standalone.
    
    Run this to test the cleaning module:
    $ python clean_data.py
    """
    # Example usage
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data('survey.csv')
    print_data_info(X_train, X_test, y_train, y_test)
    print("Feature names:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i}. {name}")