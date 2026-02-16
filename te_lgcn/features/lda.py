"""LDA topic modeling."""

from gensim import corpora
from gensim.models import LdaModel
import pandas as pd


class LDAExtractor:
    """
    Extract LDA topics from movie plot summaries.

    Args:
        n_topics (int): Number of topics to extract
        alpha (str): Document-topic density parameter
        passes (int): Number of passes through the corpus
        random_state (int): Random seed

    Example:
        >>> extractor = LDAExtractor(n_topics=10, passes=10)
        >>> topic_assignments = extractor.fit_transform(movie_texts, movie_ids)
    """

    def __init__(self, n_topics=10, alpha='auto', passes=10, random_state=42):
        self.n_topics = n_topics
        self.alpha = alpha
        self.passes = passes
        self.random_state = random_state
        self.model = None
        self.dictionary = None

    def fit_transform(self, texts, doc_ids, threshold=0.1):
        """
        Train LDA and extract topic assignments.

        Args:
            texts (list): List of preprocessed document texts (tokenized)
            doc_ids (list): List of document IDs
            threshold (float): Minimum probability to assign a topic

        Returns:
            pd.DataFrame: DataFrame with columns ['movie_id', 'topic_id', 'probability']
        """
        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        # Train LDA
        self.model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            alpha=self.alpha,
            passes=self.passes,
            random_state=self.random_state
        )

        # Extract topic assignments
        topic_assignments = []
        for doc_id, doc_bow in zip(doc_ids, corpus):
            topics = self.model.get_document_topics(doc_bow)
            for topic_id, prob in topics:
                if prob >= threshold:
                    topic_assignments.append({
                        'movie_id': doc_id,
                        'topic_id': topic_id,
                        'probability': prob
                    })

        return pd.DataFrame(topic_assignments)

    def save(self, model_path, dict_path):
        """Save LDA model and dictionary."""
        if self.model:
            self.model.save(model_path)
        if self.dictionary:
            self.dictionary.save(dict_path)

    def load(self, model_path, dict_path):
        """Load LDA model and dictionary."""
        self.model = LdaModel.load(model_path)
        self.dictionary = corpora.Dictionary.load(dict_path)
