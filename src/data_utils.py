from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import pandas as pd


def get_data_splits():
    # 1. Load data
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # 2. First split: Separate out the Test set (20%)
    # This creates 'train_val' which was missing in your error!
    train_val, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Second split: Split the remaining 80% into Train (60%) and Val (20%)
    # (0.25 of 0.8 is 0.2)
    train_df, val_df = train_test_split(train_val, test_size=0.25, random_state=42)

    return train_df, val_df, test_df