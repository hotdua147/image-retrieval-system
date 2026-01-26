import torch
import numpy as np
from tqdm import tqdm
from src.loaders.hf_dataset import load_flickr30k
from src.models.clip_wrapper import CLIPEngine

def evaluate_standard():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPEngine(device=device)
    dataset = load_flickr30k()
    
    # Lấy 1,000 mẫu cuối cùng làm tập Test chuẩn
    test_samples = 1000
    test_data = dataset.select(range(len(dataset) - test_samples, len(dataset)))

    # 1. Trích xuất Image Embeddings cho 1,000 ảnh này
    print(f"[INFO] Indexing {test_samples} test images...")
    image_list = [img.convert("RGB") for img in test_data["image"]]
    with torch.no_grad():
        image_embeddings = model.encode_images(image_list).cpu().numpy()

    # 2. Chuẩn bị FAISS tạm thời chỉ cho 1,000 ảnh này
    import faiss
    dimension = 512
    temp_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(image_embeddings)
    temp_index.add(image_embeddings)

    # 3. Đánh giá Recall
    k_values = [1, 5, 10]
    counters = {k: 0 for k in k_values}

    print(f"[INFO] Evaluating 1,000 queries against 1,000 images...")
    for i in tqdm(range(test_samples)):
        query_text = test_data[i]['caption'][0]
        gt_index = i

        with torch.no_grad():
            text_vec = model.encode_text([query_text]).cpu().numpy()
        
        faiss.normalize_L2(text_vec)
        scores, indices = temp_index.search(text_vec, max(k_values))

        for k in k_values:
            if gt_index in indices[0][:k]:
                counters[k] += 1

    print("\n--- STANDARD BENCHMARK RESULTS (1K) ---")
    for k in k_values:
        recall = (counters[k] / test_samples) * 100
        print(f"Recall@{k}: {recall:.2f}%")

if __name__ == "__main__":
    evaluate_standard()