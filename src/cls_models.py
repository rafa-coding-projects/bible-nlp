from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SearchResult:
    """Container for search results with metadata."""

    text: str
    source: str  # e.g., "Bible", "Catechism", "St. Augustine"
    reference: str  # e.g., "John 3:16", "CCC 2052"
    relevance_score: float
    positivity_score: float
    metadata: Dict = None


@dataclass
class GuidanceOutput:
    """Container for complete guidance with text and optional image."""

    query: str
    results: List[SearchResult]
    guidance_text: str
    image_url: Optional[str] = None
    image_prompt: Optional[str] = None


class DocumentSource(ABC):
    """Abstract base class for document sources."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this document source."""
        pass

    @abstractmethod
    def get_embedding_model(self):
        """Return the embedding model used by this source."""
        pass

    @abstractmethod
    def get_embeddings(self) -> np.ndarray:
        """Return the embeddings used by this source."""
        pass

    @abstractmethod
    def format_result(self, index: int) -> SearchResult:
        """Format a search result given its index."""
        pass


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def search(
        self, sources: List[DocumentSource], queries: List[str], top_k: int
    ) -> List[SearchResult]:
        """
        Execute search across multiple sources with multiple queries.

        Args:
            sources: List of document sources to search
            queries: List of query variations
            top_k: Total number of results to return

        Returns:
            Ranked list of SearchResult objects
        """
        pass


class SearchStrategyDecorator(SearchStrategy):
    """Base class for search strategy decorators."""

    def __init__(self, wrapped_strategy: SearchStrategy):
        self.wrapped_strategy = wrapped_strategy

    def set_wrapped_strategy(cls, strategy: SearchStrategy) -> SearchStrategy:
        """Set the wrapped search strategy."""
        cls.wrapped_strategy = strategy
        return cls


class GuidanceGenerator(ABC):
    """Abstract base class for guidance generation."""

    @abstractmethod
    def generate(self, query: str, results: List[SearchResult]) -> GuidanceOutput:
        """Generate guidance from search results."""
        pass


class QueryReformulator(ABC):
    """Abstract base class for query reformulation."""

    @abstractmethod
    def reformulate(self, query: str) -> List[str]:
        """
        Reformulate a query into one or more search queries.

        Args:
            query: Original user query

        Returns:
            List of reformulated queries (includes original)
        """
        pass


class NoReformulator(QueryReformulator):
    """Pass-through reformulator (no changes)."""

    def reformulate(self, query: str) -> List[str]:
        return [query]
