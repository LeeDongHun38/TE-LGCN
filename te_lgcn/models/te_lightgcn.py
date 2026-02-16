"""
Topic-Enhanced LightGCN (TE-LGCN)

This module implements the TE-LGCN model which combines:
1. Semantic initialization via Doc2Vec embeddings
2. Structural expansion via LDA topic nodes
3. Content consistency loss to preserve semantic meaning
"""

import torch
import torch.nn as nn


class TELightGCN(nn.Module):
    """
    Topic-Enhanced LightGCN for Recommendation.

    Extends LightGCN with:
    - Doc2Vec semantic initialization for item embeddings
    - LDA topic nodes in the heterogeneous graph
    - Content consistency regularization

    Args:
        n_users (int): Number of users
        n_items (int): Number of items
        n_topics (int): Number of LDA topics
        dim (int): Embedding dimension (must match Doc2Vec dimension)
        layers (int): Number of graph convolution layers
        A_hat (torch.sparse.FloatTensor): Normalized heterogeneous adjacency matrix
        doc2vec_weights (torch.Tensor, optional): Pre-trained Doc2Vec embeddings (n_items, dim)

    Example:
        >>> # With Doc2Vec initialization
        >>> model = TELightGCN(
        ...     n_users=670, n_items=3485, n_topics=10, dim=64, layers=3,
        ...     A_hat=hetero_adj_matrix, doc2vec_weights=doc2vec_emb
        ... )
        >>>
        >>> # Without Doc2Vec (random initialization)
        >>> model = TELightGCN(
        ...     n_users=670, n_items=3485, n_topics=10, dim=64, layers=3,
        ...     A_hat=hetero_adj_matrix, doc2vec_weights=None
        ... )
    """

    def __init__(self, n_users, n_items, n_topics, dim, layers, A_hat, doc2vec_weights=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_topics = n_topics
        self.dim = dim
        self.layers = layers
        self.A_hat = A_hat

        # 1. User Embedding (Random initialization)
        self.user_emb = nn.Embedding(n_users, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)

        # 2. Item Embedding (Doc2Vec initialization if available)
        if doc2vec_weights is not None:
            # Semantic initialization with Doc2Vec
            self.item_emb = nn.Embedding.from_pretrained(doc2vec_weights, freeze=False)
            # Store fixed Doc2Vec vectors for content consistency loss
            self.register_buffer('fixed_doc2vec', doc2vec_weights.clone().detach())
            print("   ✅ Item embeddings initialized with Doc2Vec (trainable)")
        else:
            # Fallback: Random initialization
            self.item_emb = nn.Embedding(n_items, dim)
            nn.init.normal_(self.item_emb.weight, std=0.1)
            self.register_buffer('fixed_doc2vec', None)
            print("   ⚠️  Item embeddings using random initialization")

        # 3. Topic Embedding (Random initialization)
        self.topic_emb = nn.Embedding(n_topics, dim)
        nn.init.normal_(self.topic_emb.weight, std=0.1)

    def get_all_embeddings(self):
        """
        Perform graph convolution on heterogeneous graph.

        Returns:
            tuple: (user_embeddings, item_embeddings)
                - user_embeddings: (n_users, dim)
                - item_embeddings: (n_items, dim)

        Note:
            Topic embeddings are used internally for message passing but not
            returned as they are not used in final scoring.
        """
        users = self.user_emb.weight
        items = self.item_emb.weight
        topics = self.topic_emb.weight

        # Concatenate all node embeddings: [users | items | topics]
        all_emb = torch.cat([users, items, topics], dim=0)
        embs = [all_emb]

        # Graph convolution through heterogeneous graph
        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.A_hat, all_emb)
            embs.append(all_emb)

        # Layer aggregation: average across all layers
        out = torch.stack(embs, dim=0).mean(dim=0)

        # Split back into user, item, topic embeddings
        users_final = out[:self.n_users]
        items_final = out[self.n_users : self.n_users + self.n_items]
        # topics_final is used internally but not returned

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

    def get_content_loss(self, item_indices):
        """
        Calculate content consistency loss for given items.

        Args:
            item_indices (torch.Tensor): Item indices to calculate loss for

        Returns:
            torch.Tensor: Content consistency loss (scalar)
        """
        if self.fixed_doc2vec is None:
            return torch.tensor(0.0, device=self.item_emb.weight.device)

        # Get current item embeddings (before graph convolution)
        current_emb = self.item_emb.weight[item_indices]
        # Get fixed Doc2Vec embeddings
        fixed_emb = self.fixed_doc2vec[item_indices]

        # L2 distance between current and fixed embeddings
        loss = (current_emb - fixed_emb).norm(2).pow(2) / item_indices.size(0)
        return loss
