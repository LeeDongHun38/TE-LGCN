"""PyTorch Dataset for recommendation data."""

import torch
from torch.utils.data import Dataset


class RecommendationDataset(Dataset):
    """
    Dataset for recommendation training with negative sampling.

    Args:
        user_pos_items (dict): Mapping from user_id to set of positive item IDs
        user_neg_items (dict): Mapping from user_id to set of negative item IDs (optional)
        n_items (int): Total number of items
        num_samples (int): Number of samples to generate per epoch

    Example:
        >>> dataset = RecommendationDataset(user_pos_items, user_neg_items, n_items=3485)
        >>> loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    """

    def __init__(self, user_pos_items, user_neg_items=None, n_items=None, num_samples=None):
        self.user_pos_items = user_pos_items
        self.user_neg_items = user_neg_items
        self.n_items = n_items
        self.users = list(user_pos_items.keys())

        if num_samples is None:
            # Default: one positive sample per user
            self.num_samples = len(self.users)
        else:
            self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get one training sample.

        Returns:
            tuple: (user_id, positive_item_id, negative_item_id)
        """
        # Sample a user
        user = self.users[idx % len(self.users)]

        # Sample a positive item for this user
        pos_items = list(self.user_pos_items[user])
        pos_item = pos_items[torch.randint(len(pos_items), (1,)).item()]

        # Sample a negative item
        if self.user_neg_items and user in self.user_neg_items:
            # Hard negative sampling
            neg_items = list(self.user_neg_items[user])
            if len(neg_items) > 0:
                neg_item = neg_items[torch.randint(len(neg_items), (1,)).item()]
            else:
                # Fallback to random sampling
                neg_item = torch.randint(self.n_items, (1,)).item()
                while neg_item in self.user_pos_items[user]:
                    neg_item = torch.randint(self.n_items, (1,)).item()
        else:
            # Random negative sampling
            neg_item = torch.randint(self.n_items, (1,)).item()
            while neg_item in self.user_pos_items[user]:
                neg_item = torch.randint(self.n_items, (1,)).item()

        return user, pos_item, neg_item
