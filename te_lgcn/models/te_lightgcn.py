"""Topic-Enhanced LightGCN with Doc2Vec initialization and LDA topic nodes."""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TELightGCN(nn.Module):
    """TE-LGCN: LightGCN with semantic initialization and topic nodes."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_topics: int,
        dim: int,
        layers: int,
        A_hat: torch.sparse.FloatTensor,
        doc2vec_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_topics = n_topics
        self.dim = dim
        self.layers = layers
        self.A_hat = A_hat

        self.user_emb = nn.Embedding(n_users, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)

        if doc2vec_weights is not None:
            self.item_emb = nn.Embedding.from_pretrained(doc2vec_weights, freeze=False)
            self.register_buffer('fixed_doc2vec', doc2vec_weights.clone().detach())
            logger.info("Item embeddings initialized with Doc2Vec (trainable)")
        else:
            self.item_emb = nn.Embedding(n_items, dim)
            nn.init.normal_(self.item_emb.weight, std=0.1)
            self.register_buffer('fixed_doc2vec', None)
            logger.warning("Item embeddings using random initialization")

        self.topic_emb = nn.Embedding(n_topics, dim)
        nn.init.normal_(self.topic_emb.weight, std=0.1)

    def get_all_embeddings(self):
        """Perform graph convolution and return final user/item embeddings."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight, self.topic_emb.weight], dim=0)
        embs = [all_emb]

        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.A_hat, all_emb)
            embs.append(all_emb)

        out = torch.stack(embs, dim=0).mean(dim=0)
        users_final = out[:self.n_users]
        items_final = out[self.n_users : self.n_users + self.n_items]

        return users_final, items_final

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Compute user-item scores."""
        users_emb, items_emb = self.get_all_embeddings()
        return (users_emb[users] * items_emb[items]).sum(1)
