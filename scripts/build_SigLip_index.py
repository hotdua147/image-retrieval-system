import torch
import numpy as np
import faiss
from tqdm import tqdm
from src.models.siglip_wrapper import SigLIPEngine
from src.loaders.hf_dataset import load_flickr30k

def build_siglip_index():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = "data/embeddings/siglip_flickr30k.index"
    
    # 1. Khởi tạo SigLIP (768-d)
    model = SigLIPEngine(model_id="google/siglip-base-patch16-224", device=device)
    dataset = load_flickr30k()
    
    # 2. Khởi tạo FAISS Index với Inner Product (IP)
    # Vì vector đã được chuẩn hóa L2 trong Engine, IP tương đương với Cosine Similarity
    dimension = 768 
    index = faiss.IndexFlatIP(dimension)
    
    print(f"[INFO] Bắt đầu trích xuất đặc trưng cho {len(dataset)} ảnh...")
    
    batch_size = 32
    all_embeddings = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = [dataset[j]['image'] for j in range(i, min(i + batch_size, len(dataset)))]
        
        # Trích xuất vector (đã được L2 normalized trong SigLIPEngine)
        with torch.no_grad():
            embeddings = model.encode_images(batch).cpu().numpy()
        
        all_embeddings.append(embeddings)

    # 3. Gộp và đưa vào FAISS
    all_embeddings = np.vstack(all_embeddings).astype('float32')
    index.add(all_embeddings)
    
    # 4. Lưu tệp tin
    faiss.write_index(index, save_path)
    print(f"[SUCCESS] Đã lưu index tại: {save_path}")

if __name__ == "__main__":
    build_siglip_index()