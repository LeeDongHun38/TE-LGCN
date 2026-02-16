"""Evaluation metrics for recommendation systems."""

import numpy as np
import torch


def calculate_recall_at_k(predicted: list, actual: set, k: int = 10) -> float:
    """Recall@K: proportion of relevant items retrieved."""
    if len(actual) == 0:
        return 0.0
    return len(set(predicted[:k]) & actual) / len(actual)


def calculate_ndcg_at_k(predicted: list, actual: set, k: int = 10) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    dcg = sum(1.0 / np.log2(i + 2) for i, item_id in enumerate(predicted[:k]) if item_id in actual)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(predicted: list, actual: set, k: int = 10) -> float:
    """Precision@K: proportion of top-k items that are relevant."""
    return len(set(predicted[:k]) & actual) / k


def calculate_hit_rate_at_k(predicted: list, actual: set, k: int = 10) -> float:
    """Hit Rate@K: whether any top-k item is relevant."""
    return 1.0 if len(set(predicted[:k]) & actual) > 0 else 0.0


def evaluate_model(model, test_data, user_pos_items: dict, k: int = 10, device: str = 'cuda'):
    """Evaluate model on test data with Recall@K, NDCG@K, Precision@K, Hit@K."""
    model.eval()
    hits, ndcg, precision, recall = 0.0, 0.0, 0.0, 0.0
    total_users = 0

    with torch.no_grad():
        user_final, item_final = model.get_all_embeddings()
        user_final = user_final.to(device)
        item_final = item_final.to(device)

        for user_id, group in test_data.groupby('user'):
            total_users += 1
            target_items = set(group['item'].values)

            scores = torch.matmul(user_final[user_id], item_final.t())

            if user_id in user_pos_items:
                mask_idx = torch.tensor(list(user_pos_items[user_id]), device=device, dtype=torch.long)
                scores[mask_idx] = -1e9

            _, topk_items = torch.topk(scores, k)
            topk_items = topk_items.cpu().tolist()

            hits += calculate_hit_rate_at_k(topk_items, target_items, k)
            recall += calculate_recall_at_k(topk_items, target_items, k)
            ndcg += calculate_ndcg_at_k(topk_items, target_items, k)
            precision += calculate_precision_at_k(topk_items, target_items, k)

    return {
        'Hit': hits / total_users if total_users > 0 else 0.0,
        'Recall': recall / total_users if total_users > 0 else 0.0,
        'NDCG': ndcg / total_users if total_users > 0 else 0.0,
        'Precision': precision / total_users if total_users > 0 else 0.0,
    }
