"""Doc2Vec feature extraction."""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np


class Doc2VecExtractor:
    """
    Extract Doc2Vec embeddings from movie plot summaries.

    Args:
        vector_size (int): Dimension of embeddings (should match model dim, e.g., 64)
        window (int): Context window size
        epochs (int): Number of training epochs
        min_count (int): Minimum word frequency

    Example:
        >>> extractor = Doc2VecExtractor(vector_size=64, epochs=20)
        >>> embeddings = extractor.fit_transform(movie_texts, movie_ids)
    """

    def __init__(self, vector_size=64, window=5, epochs=20, min_count=1, workers=4, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.min_count = min_count
        self.workers = workers
        self.seed = seed
        self.model = None

    def fit_transform(self, texts, doc_ids):
        """
        Train Doc2Vec and extract embeddings.

        Args:
            texts (list): List of document texts
            doc_ids (list): List of document IDs (corresponding to texts)

        Returns:
            dict: Mapping from doc_id to embedding vector
        """
        # Create tagged documents
        documents = [
            TaggedDocument(str(text).split(), [str(doc_id)])
            for text, doc_id in zip(texts, doc_ids)
        ]

        # Train Doc2Vec
        self.model = Doc2Vec(
            documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed
        )

        # Extract embeddings
        embeddings = {}
        for doc_id in doc_ids:
            if str(doc_id) in self.model.dv:
                embeddings[doc_id] = self.model.dv[str(doc_id)]
            else:
                # Fallback to random if not found
                embeddings[doc_id] = np.random.normal(0, 0.01, self.vector_size)

        return embeddings

    def save(self, path):
        """Save Doc2Vec model."""
        if self.model:
            self.model.save(path)

    def load(self, path):
        """Load Doc2Vec model."""
        self.model = Doc2Vec.load(path)
