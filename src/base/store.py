from abc import ABC, abstractmethod
import numpy as np

class VectorStore(ABC):
    """
    Abstract Base Class for Vector Database management.
    Ensures consistent methods for adding, saving, and searching vectors.
    """

    @abstractmethod
    def add_vectors(self, vectors: np.ndarray):
        """Add feature vectors to the index."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int):
        """Search for the most similar vectors."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Persist the index to disk."""
        pass