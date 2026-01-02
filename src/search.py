from typing import List

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from cls_models import (
    DocumentSource,
    SearchResult,
    SearchStrategy,
    SearchStrategyDecorator,
)

# Re-rank with cross-encoder (better at context)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")


class CosineSearchStrategy(SearchStrategy):
    """Simple cosine similarity search."""

    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):

        self.model = SentenceTransformer(model_name) if model_name else model

    def search(
        self, sources: List[DocumentSource], queries: List[str], top_k: int = 5
    ) -> List[SearchResult]:

        all_results = []
        # Batch encode all queries at once for GPU efficiency
        query_vecs = self.model.encode(queries, batch_size=32, show_progress_bar=False)
        
        for query_vec in query_vecs:
            for source in sources:
                verse_emb_norm = source.get_embeddings() / np.linalg.norm(
                    source.get_embeddings(), axis=1, keepdims=True
                )
                query_vec_norm = query_vec / np.linalg.norm(query_vec)

                similarities = np.dot(verse_emb_norm, query_vec_norm)
                top_k_idx = np.argsort(similarities)[-top_k:][::-1]
                for idx in top_k_idx:
                    all_results.extend([source.format_result(idx)])

        # Deduplicate and sort by relevance
        unique_results = {r.text: r for r in all_results}
        sorted_results = sorted(
            unique_results.values(), key=lambda x: x.relevance_score, reverse=True
        )
        return sorted_results[:top_k]


class HybridSearchStrategy(SearchStrategy):
    """Combines multiple strategies with cross-encoder reranking."""

    def __init__(self, use_diversity: bool = True, cross_encoder=None):

        self.cross_encoder = (
            cross_encoder
            if cross_encoder
            else CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self.use_diversity = use_diversity

    def search(
        self, sources: List[DocumentSource], queries: List[str], top_k: int
    ) -> List[SearchResult]:
        # Get candidates
        base_strategy = CosineSearchStrategy()
        candidates = base_strategy.search(sources, queries, top_k * 3)

        # Rerank with cross-encoder (use first query as reference)
        main_query = queries[0]
        pairs = [[main_query, r.text] for r in candidates]
        scores = self.cross_encoder.predict(pairs)

        for result, score in zip(candidates, scores):
            result.relevance_score = float(score)

        # Sort by reranked scores
        sorted_results = sorted(
            candidates, key=lambda x: x.relevance_score, reverse=True
        )

        return sorted_results[:top_k]


class SentimentFilteredSearchDecorator(SearchStrategyDecorator):
    """Search with sentiment filtering for positive results."""

    def __init__(
        self, min_positivity: float = 0.6, wrapped_strategy: SearchStrategy = None
    ):
        import torch
        from transformers import pipeline

        self.min_positivity = min_positivity
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1,
        )
        self.wrapped_strategy = wrapped_strategy

    def search(
        self, sources: List[DocumentSource], queries: List[str], top_k: int
    ) -> List[SearchResult]:
        # First, get candidates
        candidates = self.wrapped_strategy.search(sources, queries, top_k * 3)

        if len(candidates) <= top_k:
            return candidates

        # Score positivity - batch process for efficiency
        texts = [r.text[:512] for r in candidates]
        sentiment_results = self.sentiment_analyzer(texts, batch_size=32)
        
        for result, sentiment in zip(candidates, sentiment_results):
            if sentiment["label"] == "POSITIVE":
                result.positivity_score = sentiment["score"]
            else:
                result.positivity_score = 1.0 - sentiment["score"]

        # Filter and combine scores
        filtered = [r for r in candidates if r.positivity_score >= self.min_positivity]

        if len(filtered) < top_k:
            filtered = candidates  # Use all if not enough positive ones

        # Combined score: 60% relevance + 40% positivity
        for result in filtered:
            result.relevance_score = (
                0.6 * result.relevance_score + 0.4 * result.positivity_score
            )

        # Deduplicate and sort
        unique_results = {r.text: r for r in filtered}
        sorted_results = sorted(
            unique_results.values(), key=lambda x: x.relevance_score, reverse=True
        )
        return sorted_results[:top_k]


class MMRSearchDecorator(SearchStrategyDecorator):
    """
    Decorator that applies MMR diversity to any search strategy.

    Usage:
        base_strategy = SentimentSearchStrategy()
        diverse_strategy = MMRSearchDecorator(base_strategy, diversity_penalty=0.3)
        results = diverse_strategy.search(sources, queries, top_k=5)
    """

    def __init__(
        self,
        wrapped_strategy: SearchStrategy = None,
        diversity_penalty: float = 0.3,
        embedding_model=None,
        cross_encoder_model=None,
    ):
        """
        Args:
            wrapped_strategy: The base search strategy to decorate
            diversity_penalty: Lambda parameter for MMR (0=pure relevance, 1=pure diversity)
            embedding_model: Model to compute embeddings if not in results
        """
        self.wrapped_strategy = wrapped_strategy
        self.diversity_penalty = diversity_penalty
        self.embedding_model = model if embedding_model is None else embedding_model
        self.cross_encoder = (
            cross_encoder if cross_encoder_model is None else cross_encoder_model
        )

    def search(self, sources, queries: List[str], top_k: int) -> List[SearchResult]:
        """Apply MMR diversity to wrapped strategy's results."""

        # Get candidates from wrapped strategy (get more than needed)
        candidates = self.wrapped_strategy.search(sources, queries, 20 * 3)

        if len(candidates) <= top_k:
            return candidates  # Not enough results to diversify

        # Apply MMR
        return self._apply_mmr(queries, candidates, top_k)

    def _apply_mmr(
        self, query, candidates: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Apply Maximal Marginal Relevance algorithm."""

        verses_list = [result.text for result in candidates]

        # Use first one
        query = query[0]

        # Score by relevance and positivity
        pairs = [[query, verse] for verse in verses_list]
        relevance_scores = self.cross_encoder.predict(pairs)
        for result, score in zip(candidates, relevance_scores):
            result.relevance_score = float(score)

        # Normalize embeddings
        embeddings = self.embedding_model.encode(verses_list)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # MMR algorithm
        selected_indices = []
        selected_embeddings = []
        available_indices = list(range(len(candidates)))

        # Select first result (highest relevance)
        first_idx = np.argmax([c.relevance_score for c in candidates])
        selected_indices.append(first_idx)
        selected_embeddings.append(embeddings_norm[first_idx])
        available_indices.remove(first_idx)

        # Iteratively select diverse results
        while len(selected_indices) < top_k and available_indices:
            mmr_scores = []

            for idx in available_indices:

                relevance = candidates[idx].relevance_score

                # Max similarity to already selected results
                if selected_embeddings:
                    similarities = [
                        np.dot(embeddings_norm[idx], selected_emb)
                        for selected_emb in selected_embeddings
                    ]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0

                # MMR score: balance relevance and diversity
                mmr_score = relevance - self.diversity_penalty * max_similarity
                mmr_scores.append(mmr_score)

            # Select best MMR score
            best_mmr_idx = np.argmax(mmr_scores)
            selected_idx = available_indices[best_mmr_idx]

            selected_indices.append(selected_idx)
            selected_embeddings.append(embeddings_norm[selected_idx])
            available_indices.remove(selected_idx)

        return [candidates[idx] for idx in selected_indices]