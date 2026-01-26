import faiss
import numpy as np
import os
from src.base.store import VectorStore

class FAISSIndex(VectorStore):
    """
    FAISS implementation for efficient similarity search.
    Optimized for Inner Product (Cosine Similarity after L2 normalization).
    """

    def __init__(self, dimension: int, index_path: str = "data/embeddings/flickr30k.index"):
        """
        Initialize the FAISS Index.
        
        Args:
            dimension (int): The size of the embedding vectors (e.g., 512 for CLIP-ViT-B/32).
            index_path (str): File path to save/load the index.
        """
        self.dimension = dimension
        self.index_path = index_path
        
        # Use IndexFlatIP (Inner Product) for Cosine Similarity
        # Note: Input vectors must be L2-normalised before adding/searching
        self.index = faiss.IndexFlatIP(dimension)
        print(f"[INFO] Initialized FAISS IndexFlatIP with dimension {dimension}")

    def add_vectors(self, vectors: np.ndarray):
        """
        Add embeddings to the FAISS index.
        
        Args:
            vectors (np.ndarray): Array of shape (N, dimension) and type float32.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)

        # Ensure data type is float32 (required by FAISS)
        vectors = vectors.astype('float32')
        
        # FAISS normalization helper to ensure unit length
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        print(f"[INFO] Successfully added {vectors.shape[0]} vectors to the index.")

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Perform a nearest neighbor search.
        
        Args:
            query_vector (np.ndarray): The encoded text/image query vector.
            top_k (int): Number of nearest neighbors to retrieve.
            
        Returns:
            Tuple of (scores, indices).
        """
        # Ensure query is 2D and float32
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32')
        
        # Normalize the query vector for accurate Cosine Similarity
        faiss.normalize_L2(query_vector)
        
        # Search the index
        scores, indices = self.index.search(query_vector, top_k)
        return scores[0], indices[0]

    def save(self, path: str = None):
        """
        Save the current index to a file. 
        Managed by Git LFS in the project structure.
        """
        target_path = path if path else self.index_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        faiss.write_index(self.index, target_path)
        print(f"[INFO] Index saved to {target_path}")

    def load(self, path: str = None):
        """
        Load an existing FAISS index from disk.
        """
        target_path = path if path else self.index_path
        if os.path.exists(target_path):
            self.index = faiss.read_index(target_path)
            print(f"[INFO] Index loaded from {target_path}")
        else:
            print(f"[WARNING] No index found at {target_path}")