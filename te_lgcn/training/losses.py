"""Loss functions for TE-LGCN training."""

import torch
import torch.nn.functional as F


def bpr_loss(user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor) -> torch.Tensor:
    """BPR loss: -log(sigmoid(pos_score - neg_score))."""
    pos_scores = (user_emb * pos_item_emb).sum(1)
    neg_scores = (user_emb * neg_item_emb).sum(1)
    return F.softplus(neg_scores - pos_scores).mean()


def content_consistency_loss(learned_emb: torch.Tensor, fixed_emb: torch.Tensor) -> torch.Tensor:
    """L2 distance between learned embeddings and fixed Doc2Vec embeddings."""
    if fixed_emb is None:
        return torch.tensor(0.0, device=learned_emb.device)
    return (learned_emb - fixed_emb).pow(2).sum() / learned_emb.size(0)


def combined_loss(
    user_final: torch.Tensor,
    pos_item_final: torch.Tensor,
    neg_item_final: torch.Tensor,
    user_init: torch.Tensor,
    pos_item_init: torch.Tensor,
    neg_item_init: torch.Tensor,
    topic_init: torch.Tensor,
    fixed_doc2vec: torch.Tensor,
    lambda1: float = 1e-5,
    lambda2: float = 1e-3,
) -> torch.Tensor:
    """Combined TE-LGCN loss: L_BPR + λ1*L_reg + λ2*L_content."""
    loss_bpr = bpr_loss(user_final, pos_item_final, neg_item_final)

    # L2 regularization with correct scaling
    batch_size = user_init.size(0)
    user_item_reg = (user_init.pow(2).sum() + pos_item_init.pow(2).sum() + neg_item_init.pow(2).sum()) / batch_size
    topic_reg = topic_init.pow(2).sum() / topic_init.size(0) if topic_init.size(0) > 1 else 0.0
    reg_loss = 0.5 * (user_item_reg + topic_reg)

    content_loss = content_consistency_loss(pos_item_init, fixed_doc2vec)

    return loss_bpr + lambda1 * reg_loss + lambda2 * content_loss
