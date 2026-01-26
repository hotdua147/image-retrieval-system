import torch
import numpy as np
from src.loaders.hf_dataset import load_flickr30k
from src.models.clip_wrapper import CLIPEngine
from src.engine.faiss_index import FAISSIndex

def run_sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load các thành phần
    print("[INFO] Loading components for Sanity Check...")
    model = CLIPEngine(device=device)
    vector_store = FAISSIndex(dimension=512)
    vector_store.load() # Load tệp .index hiện tại
    dataset = load_flickr30k()

    # 2. Chọn một mẫu thử ngẫu nhiên (ví dụ ảnh số 500)
    test_idx = 500 
    sample = dataset[test_idx]
    image = sample['image'].convert("RGB")
    
    print(f"\n[STEP 1] Testing Image at Index: {test_idx}")
    print(f"[STEP 2] Encoding image and searching...")

    # 3. Trích xuất vector và tìm kiếm
    with torch.no_grad():
        query_vec = model.encode_images([image]).cpu().numpy()
    
    scores, indices = vector_store.search(query_vec, top_k=5)

    # 4. Phân tích kết quả
    top_1_idx = int(indices[0])
    top_1_score = scores[0]

    print("\n--- SANITY CHECK RESULTS ---")
    print(f"Expected Index: {test_idx}")
    print(f"Top 1 Result Index: {top_1_idx}")
    print(f"Top 1 Similarity Score: {top_1_score:.6f}")

    if top_1_idx == test_idx:
        if top_1_score > 0.99:
            print("\n✅ STATUS: SUCCESS!")
            print("Hệ thống đồng bộ hoàn hảo. Nếu Recall vẫn thấp, vấn đề nằm ở ngữ nghĩa CLIP.")
        else:
            print("\n⚠️ STATUS: ALIGNMENT OK, BUT LOW SCORE.")
            print("Thứ tự Index đúng nhưng điểm số thấp. Kiểm tra lại Normalization.")
    else:
        print("\n❌ STATUS: FAILED (INDEX MISMATCH)!")
        print(f"Hệ thống bị lệch Index. Ảnh số {test_idx} đang bị hiểu nhầm thành Index {top_1_idx}.")
        print("NGUYÊN NHÂN: Quá trình build_index bị lỗi hoặc bỏ qua (skip) một số batch.")

if __name__ == "__main__":
    run_sanity_check()