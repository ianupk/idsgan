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
    
    def transform(df):
        encoded_cats = encoder.transform(df[cat_cols])
        scaled_nums = scaler.transform(df[num_cols])
        
        result = []
        cat_idx = 0
        num_idx = 0
        
        for i in range(len(feature_utils.FEATURE_NAMES)):
            if i in cat_indices:
                cat_name = [k for k, v in feature_utils.CATEGORICAL_FEATURES.items() if v['index'] == i][0]
                num_vals = len(feature_utils.CATEGORICAL_FEATURES[cat_name]['values'])
                result.append(encoded_cats[:, cat_idx:cat_idx+num_vals])
                cat_idx += num_vals
            else:
                result.append(scaled_nums[:, num_idx:num_idx+1])
                num_idx += 1
                
        return np.hstack(result)
        
    train_X = transform(train_df)
    test_X = transform(test_df)
    return train_X, test_X, scaler, encoder

def prepare_datasets():
    """
    Main preprocessing pipeline.
    """
    print("Loading raw data...")
    train_df = load_raw_data(config.TRAIN_FILE)
    test_df = load_raw_data(config.TEST_FILE)
    
    print("Preprocessing features...")
    train_X, test_X, scaler, encoder = preprocess_features(train_df, test_df)
    
    train_y = (train_df['category'] != 'Normal').astype(int).values
    test_y = (test_df['category'] != 'Normal').astype(int).values
    
    print("Splitting KDDTrain+ into two halves...")
    indices = np.arange(len(train_df))
    idx_half1, idx_half2 = train_test_split(
        indices, 
        test_size=0.5, 
        random_state=config.RANDOM_SEED, 
        stratify=train_df['category']
    )
    
    ids_train_X = train_X[idx_half1]
    ids_train_y = train_y[idx_half1]
    
    half2_X = train_X[idx_half2]
    half2_cat = train_df.iloc[idx_half2]['category'].values
    
    gan_normal_X = half2_X[half2_cat == 'Normal']
    
    gan_attack_X = {}
    gan_attack_categories = {}
    
    # We will take attack records for GAN from half2
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
        'ids_train_y': ids_train_y,
        'gan_normal_X': gan_normal_X,
        'gan_attack_X': gan_attack_X,
        'gan_attack_categories': gan_attack_categories,
        'test_X': test_X,
        'test_y': test_y,
        'test_categories': test_df['category'].values,
        'test_attack_X': test_attack_X,
        'scaler': scaler,
        'encoder': encoder
    }
    
    return data_dict

def save_preprocessed(data_dict, path):
    """
    Save the preprocessed data dictionary using numpy.
    We separate out the scaler and encoder since np.savez_compressed doesn't handle objects well,
    but we can serialize them using joblib or save as an object array if needed.
    For simplicity, we will save arrays in npz and use joblib for models.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    npz_path = f"{path}.npz"
    models_path = f"{path}_models.joblib"
    
    # Extract arrays
    arrays_to_save = {
        'ids_train_X': data_dict['ids_train_X'],
        'ids_train_y': data_dict['ids_train_y'],
        'gan_normal_X': data_dict['gan_normal_X'],
        'test_X': data_dict['test_X'],
        'test_y': data_dict['test_y'],
        'test_categories': data_dict['test_categories']
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
        'ids_train_y': data['ids_train_y'],
        'gan_normal_X': data['gan_normal_X'],
        'test_X': data['test_X'],
        'test_y': data['test_y'],
        'test_categories': data['test_categories'],
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
    print(f"GAN Normal Shape: {data_dict['gan_normal_X'].shape}")
    print("GAN Attack Shapes:")
    for cat, arr in data_dict['gan_attack_X'].items():
        print(f"  {cat}: {arr.shape}")
    print(f"Test Shape: {data_dict['test_X'].shape}")
    
    save_path = os.path.join(config.DATA_DIR, "preprocessed")
    save_preprocessed(data_dict, save_path)
