import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

from src import config
from src import feature_utils

def load_raw_data(filepath):
    """
    Load CSV data and map attack types to categories.
    """
    col_names = feature_utils.FEATURE_NAMES + ['label', 'difficulty_level']
    df = pd.read_csv(filepath, header=None, names=col_names)
    df = df.drop(columns=['difficulty_level'])
    
    # Map attack labels to categories
    df['category'] = df['label'].map(feature_utils.ATTACK_TYPE_TO_CATEGORY).fillna('Unknown')
    return df

def preprocess_features(train_df, test_df):
    """
    Preprocess categorical and numeric features.
    Returns: train_X, test_X, scaler, encoder
    """
    cat_configs = sorted(feature_utils.CATEGORICAL_FEATURES.items(), key=lambda x: x[1]['index'])
    cat_cols = [col for col, _ in cat_configs]
    cat_categories = [config_dict['values'] for _, config_dict in cat_configs]
    
    encoder = OneHotEncoder(categories=cat_categories, handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df[cat_cols])
    
    cat_indices = [c['index'] for _, c in cat_configs]
    num_cols = [col for idx, col in enumerate(feature_utils.FEATURE_NAMES) if idx not in cat_indices]
    
    scaler = MinMaxScaler()
    scaler.fit(train_df[num_cols])

    encoded_cat_names = encoder.get_feature_names_out(cat_cols)
    final_feature_names = num_cols + list(encoded_cat_names)
    
    def transform(df):
        encoded_cats = encoder.transform(df[cat_cols])
        scaled_nums = scaler.transform(df[num_cols])
        
        return np.hstack([scaled_nums, encoded_cats])
        
    train_X = transform(train_df)
    test_X = transform(test_df)
    return train_X, test_X, scaler, encoder, final_feature_names

def prepare_datasets():
    """
    Main preprocessing pipeline.
    """
    print("Loading raw data...")
    train_df = load_raw_data(config.TRAIN_FILE)
    test_df = load_raw_data(config.TEST_FILE)
    
    print("Preprocessing features...")
    train_X, test_X, scaler, encoder, feature_names = preprocess_features(train_df, test_df)

    train_y_bin = (train_df['category'].str.lower() != 'normal').astype(int).values
    test_y_bin = (test_df['category'].str.lower() != 'normal').astype(int).values
    
    train_y_multi = train_df['category'].values
    test_y_multi = test_df['category'].values
    
    print("Splitting KDDTrain+ into two halves...")
    indices = np.arange(len(train_df))
    
    category_counts = train_df['category'].value_counts()
    valid_cats = category_counts[category_counts > 1].index
    safe_stratify_col = train_df['category'].copy()
    safe_stratify_col[~safe_stratify_col.isin(valid_cats)] = 'Rare_Unknown'
    
    idx_half1, idx_half2 = train_test_split(
        indices, 
        test_size=0.5, 
        random_state=config.RANDOM_SEED, 
        stratify=safe_stratify_col
    )
    
    idx_ids_train, idx_ids_val = train_test_split(
        idx_half1,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=safe_stratify_col.iloc[idx_half1]
    )
    
    # Populate IDS splits
    ids_train_X = train_X[idx_ids_train]
    ids_train_y_bin = train_y_bin[idx_ids_train]
    ids_train_y_multi = train_y_multi[idx_ids_train]
    
    ids_val_X = train_X[idx_ids_val]
    ids_val_y_bin = train_y_bin[idx_ids_val]
    ids_val_y_multi = train_y_multi[idx_ids_val]
    
    # Populate GAN splits (Half 2)
    half2_X = train_X[idx_half2]
    half2_cat = train_df.iloc[idx_half2]['category'].values
    
    gan_normal_X = half2_X[pd.Series(half2_cat).str.lower() == 'normal']
    
    gan_attack_X = {}
    gan_attack_categories = {}
    
    for cat in config.ATTACK_CATEGORIES:
        mask = half2_cat == cat
        if np.any(mask):
            gan_attack_X[cat] = half2_X[mask]
            gan_attack_categories[cat] = half2_cat[mask]
            
    test_attack_X = {}
    for cat in config.ATTACK_CATEGORIES:
        mask = test_df['category'].values == cat
        if np.any(mask):
            test_attack_X[cat] = test_X[mask]
            
    data_dict = {
        'ids_train_X': ids_train_X,
        'ids_train_y_bin': ids_train_y_bin,
        'ids_train_y_multi': ids_train_y_multi,
        'ids_val_X': ids_val_X,
        'ids_val_y_bin': ids_val_y_bin,
        'ids_val_y_multi': ids_val_y_multi,
        'gan_normal_X': gan_normal_X,
        'gan_attack_X': gan_attack_X,
        'gan_attack_categories': gan_attack_categories,
        'test_X': test_X,
        'test_y_bin': test_y_bin,
        'test_y_multi': test_y_multi,
        'test_attack_X': test_attack_X,
        'scaler': scaler,
        'encoder': encoder,
        'feature_names': np.array(feature_names)
    }
    return data_dict

def save_preprocessed(data_dict, path):
    """
    Save the preprocessed data dictionary using numpy.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    npz_path = f"{path}.npz"
    models_path = f"{path}_models.joblib"
    
    # Extract arrays
    arrays_to_save = {
        'ids_train_X': data_dict['ids_train_X'],
        'ids_train_y_bin': data_dict['ids_train_y_bin'],
        'ids_train_y_multi': data_dict['ids_train_y_multi'],
        'ids_val_X': data_dict['ids_val_X'],
        'ids_val_y_bin': data_dict['ids_val_y_bin'],
        'ids_val_y_multi': data_dict['ids_val_y_multi'],
        'gan_normal_X': data_dict['gan_normal_X'],
        'test_X': data_dict['test_X'],
        'test_y_bin': data_dict['test_y_bin'],
        'test_y_multi': data_dict['test_y_multi'],
        'feature_names': data_dict['feature_names']
    }
    
    # Add dict items
    for cat, data in data_dict['gan_attack_X'].items():
        arrays_to_save[f'gan_attack_X_{cat}'] = data
    for cat, data in data_dict['gan_attack_categories'].items():
        arrays_to_save[f'gan_attack_categories_{cat}'] = data
    for cat, data in data_dict['test_attack_X'].items():
        arrays_to_save[f'test_attack_X_{cat}'] = data
        
    np.savez_compressed(npz_path, **arrays_to_save)
    
    # Save sklearn models
    joblib.dump({'scaler': data_dict['scaler'], 'encoder': data_dict['encoder']}, models_path)
    print(f"Saved preprocessed data to {npz_path} and models to {models_path}")

def load_preprocessed(path):
    """
    Load preprocessed data dictionary.
    """
    npz_path = f"{path}.npz"
    models_path = f"{path}_models.joblib"
    
    data = np.load(npz_path, allow_pickle=True)
    models = joblib.load(models_path)
    
    data_dict = {
        'ids_train_X': data['ids_train_X'],
        'ids_train_y_bin': data['ids_train_y_bin'],
        'ids_train_y_multi': data['ids_train_y_multi'],
        'ids_val_X': data['ids_val_X'],
        'ids_val_y_bin': data['ids_val_y_bin'],
        'ids_val_y_multi': data['ids_val_y_multi'],
        'gan_normal_X': data['gan_normal_X'],
        'test_X': data['test_X'],
        'test_y_bin': data['test_y_bin'],
        'test_y_multi': data['test_y_multi'],
        'feature_names': data['feature_names'],
        'gan_attack_X': {},
        'gan_attack_categories': {},
        'test_attack_X': {},
        'scaler': models['scaler'],
        'encoder': models['encoder']
    }
    
    for key in data.files:
        if key.startswith('gan_attack_X_'):
            cat = key.replace('gan_attack_X_', '')
            data_dict['gan_attack_X'][cat] = data[key]
        elif key.startswith('gan_attack_categories_'):
            cat = key.replace('gan_attack_categories_', '')
            data_dict['gan_attack_categories'][cat] = data[key]
        elif key.startswith('test_attack_X_'):
            cat = key.replace('test_attack_X_', '')
            data_dict['test_attack_X'][cat] = data[key]
            
    return data_dict

if __name__ == "__main__":
    if not os.path.exists(config.TRAIN_FILE) or not os.path.exists(config.TEST_FILE):
        print("Dataset not found. Please run download_data.py first.")
        exit(1)
        
    data_dict = prepare_datasets()
    print("\nDataset Statistics:")
    print(f"IDS Train Shape: {data_dict['ids_train_X'].shape}")
    print(f"IDS Validation Shape: {data_dict['ids_val_X'].shape}")
    print(f"GAN Normal Shape: {data_dict['gan_normal_X'].shape}")
    print("GAN Attack Shapes:")
    for cat, arr in data_dict['gan_attack_X'].items():
        print(f"  {cat}: {arr.shape}")
    print(f"Test Shape: {data_dict['test_X'].shape}")
    print(f"Total Features: {len(data_dict['feature_names'])}")
    
    save_path = os.path.join(config.DATA_DIR, "preprocessed")
    save_preprocessed(data_dict, save_path)