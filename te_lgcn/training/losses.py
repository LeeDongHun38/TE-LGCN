"""
Loss functions for TE-LGCN training.

Implements:
- BPR (Bayesian Personalized Ranking) loss
- Content consistency loss
- Combined loss with regularization
"""

import torch
import torch.nn.functional as F


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """
    Bayesian Personalized Ranking (BPR) loss.

    Encourages positive items to have higher scores than negative items.

    Args:
        user_emb (torch.Tensor): User embeddings (batch_size, dim)
        pos_item_emb (torch.Tensor): Positive item embeddings (batch_size, dim)
        neg_item_emb (torch.Tensor): Negative item embeddings (batch_size, dim)

    Returns:
        torch.Tensor: BPR loss (scalar)

    Reference:
        Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009
    """
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # (batch_size,)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)  # (batch_size,)

    # softplus(-x) is a smooth approximation of max(0, -x)
    # Equivalent to -log(sigmoid(pos_scores - neg_scores))
    loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))

    return loss


def content_consistency_loss(learned_emb, fixed_emb):
    """
    Content consistency loss to keep learned embeddings close to Doc2Vec.

    Args:
        learned_emb (torch.Tensor): Learned item embeddings (batch_size, dim)
        fixed_emb (torch.Tensor): Fixed Doc2Vec embeddings (batch_size, dim)

    Returns:
        torch.Tensor: Content consistency loss (scalar)
    """
    if fixed_emb is None:
        return torch.tensor(0.0, device=learned_emb.device)

    # L2 distance between learned and fixed embeddings
    loss = (learned_emb - fixed_emb).norm(2).pow(2) / learned_emb.size(0)
    return loss


def combined_loss(
    user_final,
    pos_item_final,
    neg_item_final,
    user_init,
    pos_item_init,
    neg_item_init,
    topic_init,
    fixed_doc2vec,
    lambda1=1e-5,
    lambda2=1e-3,
):
    """
    Combined TE-LGCN loss function.

    L_total = L_BPR + λ1 * L_reg + λ2 * L_content

    Args:
        user_final (torch.Tensor): Final user embeddings after GCN (batch_size, dim)
        pos_item_final (torch.Tensor): Final positive item embeddings (batch_size, dim)
        neg_item_final (torch.Tensor): Final negative item embeddings (batch_size, dim)
        user_init (torch.Tensor): Initial user embeddings E^0 (batch_size, dim)
        pos_item_init (torch.Tensor): Initial positive item embeddings E^0 (batch_size, dim)
        neg_item_init (torch.Tensor): Initial negative item embeddings E^0 (batch_size, dim)
        topic_init (torch.Tensor): All topic embeddings E^0 (n_topics, dim)
        fixed_doc2vec (torch.Tensor): Fixed Doc2Vec embeddings for positive items (batch_size, dim)
        lambda1 (float): Weight for L2 regularization
        lambda2 (float): Weight for content consistency loss

    Returns:
        torch.Tensor: Combined loss (scalar)

    Example:
        >>> loss = combined_loss(
        ...     u_final, i_pos_final, i_neg_final,
        ...     u_0, i_pos_0, i_neg_0, topic_0,
        ...     fixed_vec, lambda1=1e-5, lambda2=1e-3
        ... )
    """
    # 1. BPR Loss (Ranking Loss)
    loss_bpr = bpr_loss(user_final, pos_item_final, neg_item_final)

    # 2. L2 Regularization Loss (on initial embeddings)
    # Regularize all node types: user, item, topic
    reg_loss = 0.5 * (
        user_init.norm(2).pow(2)
        + pos_item_init.norm(2).pow(2)
        + neg_item_init.norm(2).pow(2)
        + topic_init.norm(2).pow(2)
    ) / user_init.size(0)

    # 3. Content Consistency Loss
    # Keep learned item embeddings close to Doc2Vec semantic initialization
    content_loss = content_consistency_loss(pos_item_init, fixed_doc2vec)

    # Combined loss
    total_loss = loss_bpr + lambda1 * reg_loss + lambda2 * content_loss

    return total_loss
