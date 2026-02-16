"""Trainer for TE-LGCN models."""

import torch
from te_lgcn.training.losses import combined_loss


class Trainer:
    """Trainer for TE-LGCN and baseline LightGCN models."""

    def __init__(self, model, optimizer, device, lambda1=1e-5, lambda2=1e-3):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def train_epoch(self, train_loader):
        """Train for one epoch with efficient graph propagation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Compute graph propagation once per epoch (not per batch)
        with torch.no_grad():
            user_final, item_final = self.model.get_all_embeddings()

        for users, pos_items, neg_items in train_loader:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            # Lookup embeddings from pre-computed propagation
            u_final = user_final[users]
            i_pos_final = item_final[pos_items]
            i_neg_final = item_final[neg_items]

            # Initial embeddings for regularization
            u_0 = self.model.user_emb.weight[users]
            i_pos_0 = self.model.item_emb.weight[pos_items]
            i_neg_0 = self.model.item_emb.weight[neg_items]

            # Topic embeddings (if using TE-LGCN)
            if hasattr(self.model, 'topic_emb'):
                t_0 = self.model.topic_emb.weight
                fixed_vec = self.model.fixed_doc2vec[pos_items] if hasattr(self.model, 'fixed_doc2vec') and self.model.fixed_doc2vec is not None else None
            else:
                t_0 = torch.zeros(1, self.model.dim, device=self.device)
                fixed_vec = None

            # Calculate loss
            loss = combined_loss(
                u_final, i_pos_final, i_neg_final,
                u_0, i_pos_0, i_neg_0, t_0,
                fixed_vec, self.lambda1, self.lambda2
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path))
