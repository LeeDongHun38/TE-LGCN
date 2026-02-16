"""Base LightGCN implementation for collaborative filtering."""

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """Vanilla LightGCN with graph convolution for user-item bipartite graph."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int,
        layers: int,
        A_hat: torch.sparse.FloatTensor,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.layers = layers
        self.A_hat = A_hat

        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def get_all_embeddings(self):
        """Perform graph convolution and return final embeddings."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.A_hat, all_emb)
            embs.append(all_emb)

        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[:self.n_users], out[self.n_users:]

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Compute user-item scores."""
        users_emb, items_emb = self.get_all_embeddings()
        return (users_emb[users] * items_emb[items]).sum(1)
