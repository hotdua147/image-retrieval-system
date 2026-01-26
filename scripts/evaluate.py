import torch
from tqdm import tqdm
from src.loaders.hf_dataset import load_flickr30k
from src.models.clip_wrapper import CLIPEngine
from src.engine.faiss_index import FAISSIndex

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPEngine(device=device)
    vector_store = FAISSIndex(dimension=512)
    vector_store.load()
    
    # Đảm bảo split này KHỚP HOÀN TOÀN với lúc build_index
    dataset = load_flickr30k() 

    k_values = [1, 5, 10]
    counters = {k: 0 for k in k_values}
    
    # Chúng ta sẽ test trên 1000 mẫu đầu tiên để kiểm chứng nhanh
    num_samples = 1000 
    print(f"[INFO] Evaluating Recall@K on {num_samples} queries...")

    for i in tqdm(range(num_samples)):
        # Lấy caption đầu tiên của ảnh thứ i làm câu truy vấn
        query_text = dataset[i]['caption'][0] 
        
        # Ảnh gốc (Ground Truth) chính là index i trong FAISS
        gt_index = i 

        with torch.no_grad():
            query_vec = model.encode_text([query_text]).cpu().numpy()
        
        # Tìm kiếm trong FAISS
        scores, indices = vector_store.search(query_vec, top_k=max(k_values))
        
        # Kiểm tra xem gt_index có nằm trong top K không
        for k in k_values:
            if gt_index in indices[:k]:
                counters[k] += 1

    print("\n--- FINAL EVALUATION RESULTS ---")
    for k in k_values:
        recall = (counters[k] / num_samples) * 100
        print(f"Recall@{k}: {recall:.2f}%")

if __name__ == "__main__":
    evaluate()