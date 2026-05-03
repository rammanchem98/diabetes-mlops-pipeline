import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def get_data_splits():
    """
    Performs a strict 60/20/2t(df, test_size=0.2, random_state=42)

    # 2. Separate Train (60%) and Val (20%)0 split on the Diabetes dataset.
    #     Returns: train, val, test DataFrames
    #     """
    #     dataset = load_diabetes()
    #     df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    #     df['target'] = dataset.target
    #
    #     # 1. Separate Test set (20%)
    #     train_val, test = train_test_spli
    # 0.25 * 0.8 = 0.2
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    return train, val, test