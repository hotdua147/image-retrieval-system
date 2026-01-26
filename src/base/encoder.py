from abc import ABC, abstractmethod
from typing import List, Any
import torch

class BaseEncoder(ABC):
    """
    Abstract Base Class (ABC) for Vision-Language Encoders.
    Ensures that all model wrappers (CLIP, SigLIP, etc.) follow the same interface.
    """

    @abstractmethod
    def encode_images(self, images: List[Any]) -> torch.Tensor:
        """
        Abstract method to transform a list of images into feature vectors.
        
        Args:
            images (List[Any]): List of PIL images or image data.
            
        Returns:
            torch.Tensor: Normalised feature embeddings for images.
        """
        pass

    @abstractmethod
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Abstract method to transform a list of text queries into feature vectors.
        
        Args:
            text (List[str]): List of captions or search queries.
            
        Returns:
            torch.Tensor: Normalised feature embeddings for text.
        """
        pass