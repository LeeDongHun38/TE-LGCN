"""Data splitting utilities."""

import pandas as pd


def kcore_filter(df, k=10):
    """
    Apply k-core filtering to ensure all users and items have at least k interactions.

    Args:
        df (pd.DataFrame): Rating data with columns ['userId', 'movieId', ...]
        k (int): Minimum number of interactions

    Returns:
        pd.DataFrame: Filtered data
    """
    print(f"\n=== {k}-Core Filtering ===")
    print(f"Original: {len(df)} interactions")

    while True:
        start_len = len(df)

        # Filter users
        user_counts = df['userId'].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df['userId'].isin(valid_users)]

        # Filter items
        item_counts = df['movieId'].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df['movieId'].isin(valid_items)]

        end_len = len(df)

        if start_len == end_len:
            break

    print(f"After filtering: {len(df)} interactions")
    return df


def split_data(df, rating_threshold=4.0):
    """
    Split data into train/val/test using leave-one-out strategy.

    Args:
        df (pd.DataFrame): Rating data with columns ['userId', 'movieId', 'rating']
        rating_threshold (float): Minimum rating to consider as positive

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # TODO: Implement smart leave-one-out split
    # See notebooks/preprocessing/data_preparation.ipynb for reference
    raise NotImplementedError("Please refer to notebooks for implementation")
