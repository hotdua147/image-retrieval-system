import torch
import numpy as np
from tqdm import tqdm
from src.loaders.hf_dataset import load_flickr30k
from src.models.clip_wrapper import CLIPEngine
from src.engine.faiss_index import FAISSIndex

def run_indexing():
    """
    Main script to extract image embeddings and build a FAISS index.
    Process: Load Data -> Convert to RGB -> Encode with CLIP -> Store in FAISS.
    """
    
    # 1. Configuration & Hardware Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    embedding_dim = 512
    batch_size = 64
    index_save_path = "data/embeddings/flickr30k.index"
    
    print(f"[INFO] Starting indexing pipeline on device: {device}")

    # 2. Initialize Core Components
    model = CLIPEngine(model_id=model_id, device=device)
    vector_store = FAISSIndex(dimension=embedding_dim, index_path=index_save_path)

    # 3. Load Flickr30k Dataset
    dataset = load_flickr30k()
    if dataset is None:
        print("[ERROR] Failed to load dataset. Exiting...")
        return

    # 4. Feature Extraction Loop
    print(f"[INFO] Extracting features for {len(dataset)} images...")
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Indexing"):
        batch = dataset[i : i + batch_size]
        
        # --- CẬP NHẬT: Ép kiểu ảnh sang RGB ---
        # Đảm bảo mọi ảnh đều có 3 kênh màu (R, G, B) trước khi đưa vào CLIP
        raw_images = batch["image"]
        images_rgb = [img.convert("RGB") for img in raw_images]
        
        try:
            # Encode images to normalized vectors on GPU
            with torch.no_grad():
                img_embeddings = model.encode_images(images_rgb)
            
            # Convert to NumPy float32
            embeddings_np = img_embeddings.cpu().numpy().astype('float32')
            
            # Add vectors to the FAISS index
            vector_store.add_vectors(embeddings_np)
            
        except Exception as e:
            print(f"[ERROR] Batch at index {i} failed: {e}")
            continue

    # 5. Finalize and Save
    vector_store.save()
    print(f"[SUCCESS] Indexing completed. File saved at: {index_save_path}")

if __name__ == "__main__":
    run_indexing()