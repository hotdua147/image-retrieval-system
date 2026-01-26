import os
from datasets import load_dataset

def load_flickr30k(dataset_id="nlphuji/flickr30k", cache_dir="data/raw/"):
    """
    Downloads and loads the Flickr30k dataset from Hugging Face.
    This function will be called by other scripts.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    print(f"[INFO] Loading dataset '{dataset_id}'...")
    
    # Load dataset and cache it in data/raw/
    # This matches the 'Use in dataset library' guide from HF documentation
    dataset = load_dataset(dataset_id, cache_dir=cache_dir, split='test', trust_remote_code=True)
    
    return dataset

# No logic to run indexing here!