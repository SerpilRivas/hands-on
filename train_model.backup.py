# train_model.py
import pandas as pd
import numpy as np
from pycaret.classification import *
import os
import matplotlib.pyplot as plt


def generate_synthetic_data(num_samples=500):
    """Generates a synthetic dataset of phishing and benign URL features."""
    print("Generating synthetic dataset...")

    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL'
    ]

    num_phishing = num_samples // 2
    num_benign = num_samples - num_phishing

    phishing_data = {
        'having_IP_Address': np.random.choice([1, -1], num_phishing, p=[0.3, 0.7]),
        'URL_Length': np.random.choice([1, 0, -1], num_phishing, p=[0.5, 0.4, 0.1]),
        'Shortining_Service': np.random.choice([1, -1], num_phishing, p=[0.6, 0.4]),
        'having_At_Symbol': np.random.choice([1, -1], num_phishing, p=[0.4, 0.6]),
        'double_slash_redirecting': np.random.choice([1, -1], num_phishing, p=[0.3, 0.7]),
        'Prefix_Suffix': np.random.choice([1, -1], num_phishing, p=[0.7, 0.3]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_phishing, p=[0.6, 0.3, 0.1]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_phishing, p=[0.6, 0.3, 0.1]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_phishing, p=[0.5, 0.3, 0.2]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_phishing, p=[0.4, 0.4, 0.2]),
        'SFH': np.random.choice([-1, 0, 1], num_phishing, p=[0.7, 0.2, 0.1]),
        'Abnormal_URL': np.random.choice([1, -1], num_phishing, p=[0.5, 0.5])
    }

    benign_data = {
        'having_IP_Address': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'URL_Length': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.6, 0.3]),
        'Shortining_Service': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_At_Symbol': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'double_slash_redirecting': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'Prefix_Suffix': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.4, 0.5]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_benign, p=[0.05, 0.15, 0.8]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'SFH': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': np.random.choice([1, -1], num_benign, p=[0.1, 0.9])
    }

    df_phishing = pd.DataFrame(phishing_data)
    df_benign = pd.DataFrame(benign_data)

    df_phishing['label'] = 1
    df_benign['label'] = 0

    final_df = pd.concat([df_phishing, df_benign], ignore_index=True)
    return final_df.sample(frac=1).reset_index(drop=True)


def train():
    model_path = 'models/phishing_url_detector'
    plot_path = 'models/feature_importance.png'

    if os.path.exists(model_path + '.pkl'):
        print("Model and plot already exist. Skipping training.")
        return

    data = generate_synthetic_data()
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/phishing_synthetic.csv', index=False)

    print("Initializing PyCaret Setup...")
    s = setup(data, target='label', session_id=42, verbose=False)

    print("Comparing models...")
    best_model = compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

    print("Finalizing model...")
    final_model = finalize_model(best_model)

    # NEW: Plot feature importance and save it to a file
    print("Saving feature importance plot...")
    os.makedirs('models', exist_ok=True)
    plot_model(final_model, plot='feature', save=True)
    # PyCaret saves it as 'Feature Importance.png', let's rename it
    os.rename('Feature Importance.png', plot_path)

    print("Saving model...")
    save_model(final_model, model_path)
    print(f"Model and plot saved successfully.")


if __name__ == "__main__":
    train()