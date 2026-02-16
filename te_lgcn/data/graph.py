"""Graph construction utilities."""

import torch
import scipy.sparse as sp
import numpy as np


def build_graph(n_users, n_items, interactions):
    """
    Build normalized adjacency matrix for user-item bipartite graph.

    Args:
        n_users (int): Number of users
        n_items (int): Number of items
        interactions (pd.DataFrame): DataFrame with columns ['user', 'item']

    Returns:
        torch.sparse.FloatTensor: Normalized adjacency matrix

    Example:
        >>> adj_matrix = build_graph(n_users=670, n_items=3485, interactions=train_df)
    """
    # Create user-item interaction matrix
    users = interactions['user'].values
    items = interactions['item'].values

    # Build adjacency matrix
    n_nodes = n_users + n_items

    # Create edge lists: user -> item and item -> user
    user_item_edges = np.column_stack([users, items + n_users])
    item_user_edges = np.column_stack([items + n_users, users])

    edges = np.vstack([user_item_edges, item_user_edges])

    # Create sparse matrix
    adj = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_nodes, n_nodes),
        dtype=np.float32
    )

    # Normalize: D^(-1/2) * A * D^(-1/2)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    # Convert to PyTorch sparse tensor
    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)

    adj_tensor = torch.sparse.FloatTensor(indices, values, shape)

    return adj_tensor


def build_heterogeneous_graph(n_users, n_items, n_topics, user_item_edges, item_topic_edges):
    """
    Build normalized adjacency matrix for heterogeneous graph with topics.

    Args:
        n_users (int): Number of users
        n_items (int): Number of items
        n_topics (int): Number of topics
        user_item_edges (pd.DataFrame): DataFrame with columns ['user', 'item']
        item_topic_edges (pd.DataFrame): DataFrame with columns ['item', 'topic']

    Returns:
        torch.sparse.FloatTensor: Normalized heterogeneous adjacency matrix

    Example:
        >>> adj = build_heterogeneous_graph(
        ...     n_users=670, n_items=3485, n_topics=10,
        ...     user_item_edges=train_df,
        ...     item_topic_edges=topic_df
        ... )
    """
    n_nodes = n_users + n_items + n_topics

    # User-Item edges
    ui_users = user_item_edges['user'].values
    ui_items = user_item_edges['item'].values + n_users

    # Item-Topic edges
    it_items = item_topic_edges['item'].values + n_users
    it_topics = item_topic_edges['topic'].values + n_users + n_items

    # Create bidirectional edges
    edges = []
    # User <-> Item
    edges.append(np.column_stack([ui_users, ui_items]))
    edges.append(np.column_stack([ui_items, ui_users]))
    # Item <-> Topic
    edges.append(np.column_stack([it_items, it_topics]))
    edges.append(np.column_stack([it_topics, it_items]))

    edges = np.vstack(edges)

    # Create sparse matrix
    adj = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_nodes, n_nodes),
        dtype=np.float32
    )

    # Normalize
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    # Convert to PyTorch sparse tensor
    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)

    adj_tensor = torch.sparse.FloatTensor(indices, values, shape)

    return adj_tensor
