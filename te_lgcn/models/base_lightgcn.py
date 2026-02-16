"""
Base LightGCN Model

This module contains the vanilla LightGCN implementation for collaborative filtering.
Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation", SIGIR 2020.
"""

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    LightGCN: Simplified Graph Convolution Network for Recommendation.

    Uses graph convolution to propagate embeddings through the user-item
    bipartite graph without feature transformation or nonlinear activation.

    Args:
        n_users (int): Number of users
        n_items (int): Number of items
        dim (int): Embedding dimension
        layers (int): Number of graph convolution layers
        A_hat (torch.sparse.FloatTensor): Normalized adjacency matrix

    Example:
        >>> model = LightGCN(n_users=670, n_items=3485, dim=64, layers=3, A_hat=adj_matrix)
        >>> user_emb, item_emb = model.get_all_embeddings()
    """

    def __init__(self, n_users, n_items, dim, layers, A_hat):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.layers = layers
        self.A_hat = A_hat

        # User and item embeddings (randomly initialized)
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def get_all_embeddings(self):
        """
        Perform graph convolution and return final user and item embeddings.

        Returns:
            tuple: (user_embeddings, item_embeddings)
                - user_embeddings: (n_users, dim)
                - item_embeddings: (n_items, dim)
        """
        users = self.user_emb.weight
        items = self.item_emb.weight

        # Concatenate user and item embeddings
        all_emb = torch.cat([users, items], dim=0)
        embs = [all_emb]

        # Graph convolution
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.A_hat, all_emb)
            embs.append(all_emb)

        # Layer aggregation: average across all layers
        out = torch.stack(embs, dim=0).mean(dim=0)

        users_final = out[:self.n_users]
        items_final = out[self.n_users:]

        return users_final, items_final

    def forward(self, users, items):
        """
        Forward pass for computing user-item scores.

        Args:
            users (torch.Tensor): User indices (batch_size,)
            items (torch.Tensor): Item indices (batch_size,)

        Returns:
            torch.Tensor: Predicted scores (batch_size,)
        """
        users_emb, items_emb = self.get_all_embeddings()

        user_emb = users_emb[users]
        item_emb = items_emb[items]

        scores = (user_emb * item_emb).sum(dim=1)
        return scores
