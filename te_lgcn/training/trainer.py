"""
Trainer class for TE-LGCN models.

Handles training loop, validation, and model checkpointing.
"""

import torch
from te_lgcn.training.losses import combined_loss


class Trainer:
    """
    Trainer for TE-LGCN and baseline LightGCN models.

    Args:
        model: The model to train
        optimizer: PyTorch optimizer
        device: Device to train on ('cuda' or 'cpu')
        lambda1 (float): L2 regularization weight
        lambda2 (float): Content consistency loss weight

    Example:
        >>> model = TELightGCN(...)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> trainer = Trainer(model, optimizer, device='cuda', lambda1=1e-5, lambda2=1e-3)
        >>> trainer.train_epoch(train_data)
    """

    def __init__(self, model, optimizer, device, lambda1=1e-5, lambda2=1e-3):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def train_epoch(self, train_loader):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader providing (users, pos_items, neg_items) batches

        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for users, pos_items, neg_items in train_loader:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            # Forward pass
            user_final, item_final = self.model.get_all_embeddings()

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
                # Fixed Doc2Vec embeddings
                if hasattr(self.model, 'fixed_doc2vec') and self.model.fixed_doc2vec is not None:
                    fixed_vec = self.model.fixed_doc2vec[pos_items]
                else:
                    fixed_vec = None
            else:
                # For baseline LightGCN, no topics
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

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path))
