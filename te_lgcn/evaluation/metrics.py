"""
Evaluation metrics for recommendation systems.

Implements standard metrics: Recall@K, NDCG@K, Precision@K, Hit Rate@K
"""

import numpy as np
import torch


def calculate_recall_at_k(predicted, actual, k=10):
    """
    Calculate Recall@K.

    Args:
        predicted (list): List of predicted item IDs (top-k)
        actual (set): Set of actual positive item IDs
        k (int): Cutoff position

    Returns:
        float: Recall@K score
    """
    if len(actual) == 0:
        return 0.0

    num_correct = len(set(predicted[:k]) & actual)
    recall = num_correct / len(actual)
    return recall


def calculate_ndcg_at_k(predicted, actual, k=10):
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    Args:
        predicted (list): List of predicted item IDs (top-k)
        actual (set): Set of actual positive item IDs
        k (int): Cutoff position

    Returns:
        float: NDCG@K score
    """
    dcg = 0.0
    for i, item_id in enumerate(predicted[:k]):
        if item_id in actual:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0

    # Ideal DCG
    idcg = 0.0
    for i in range(min(len(actual), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def calculate_precision_at_k(predicted, actual, k=10):
    """
    Calculate Precision@K.

    Args:
        predicted (list): List of predicted item IDs (top-k)
        actual (set): Set of actual positive item IDs
        k (int): Cutoff position

    Returns:
        float: Precision@K score
    """
    num_correct = len(set(predicted[:k]) & actual)
    precision = num_correct / k
    return precision


def calculate_hit_rate_at_k(predicted, actual, k=10):
    """
    Calculate Hit Rate@K (whether any predicted item is in actual).

    Args:
        predicted (list): List of predicted item IDs (top-k)
        actual (set): Set of actual positive item IDs
        k (int): Cutoff position

    Returns:
        float: 1.0 if hit, 0.0 otherwise
    """
    hit = 1.0 if len(set(predicted[:k]) & actual) > 0 else 0.0
    return hit


def evaluate_model(model, test_data, user_pos_items, k=10, device='cuda'):
    """
    Evaluate model on test data.

    Args:
        model: Trained model (LightGCN or TELightGCN)
        test_data (pd.DataFrame): Test data with columns ['user', 'item']
        user_pos_items (dict): Dict mapping user_id to set of positive training items (to mask)
        k (int): Cutoff for metrics
        device (str): Device to run evaluation on

    Returns:
        dict: Dictionary containing 'Recall', 'NDCG', 'Precision', 'Hit' metrics

    Example:
        >>> results = evaluate_model(model, test_df, user_pos_items, k=10)
        >>> print(f"Recall@10: {results['Recall']:.4f}")
    """
    model.eval()

    hits, ndcg, precision, recall = 0, 0, 0, 0
    total_users = 0

    with torch.no_grad():
        user_final, item_final = model.get_all_embeddings()

        for user_id, group in test_data.groupby('user'):
            total_users += 1
            target_items = set(group['item'].values)

            # Calculate scores for all items
            scores = torch.matmul(user_final[user_id], item_final.t())

            # Mask training items
            if user_id in user_pos_items:
                scores[list(user_pos_items[user_id])] = -1e9

            # Get top-k items
            _, topk_items = torch.topk(scores, k)
            topk_items = topk_items.cpu().tolist()

            # Calculate metrics
            hits += calculate_hit_rate_at_k(topk_items, target_items, k)
            recall += calculate_recall_at_k(topk_items, target_items, k)
            ndcg += calculate_ndcg_at_k(topk_items, target_items, k)
            precision += calculate_precision_at_k(topk_items, target_items, k)

    # Average over all users
    results = {
        'Hit': hits / total_users if total_users > 0 else 0.0,
        'Recall': recall / total_users if total_users > 0 else 0.0,
        'NDCG': ndcg / total_users if total_users > 0 else 0.0,
        'Precision': precision / total_users if total_users > 0 else 0.0,
    }

    return results
