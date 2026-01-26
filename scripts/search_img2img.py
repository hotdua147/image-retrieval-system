import torch
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from src.models.clip_wrapper import CLIPEngine
from src.engine.faiss_index import FAISSIndex
from src.loaders.hf_dataset import load_flickr30k

def run_random_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Khởi tạo
    print("[INFO] Loading components...")
    model = CLIPEngine(device=device)
    vector_store = FAISSIndex(dimension=512)
    vector_store.load()
    dataset = load_flickr30k()

    # 2. Chọn ảnh ngẫu nhiên từ thư mục mẫu
    test_dir = "data/test_samples"
    all_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    random_file = random.choice(all_files)
    img_path = os.path.join(test_dir, random_file)
    
    print(f"[INFO] Selected Random Image: {img_path}")
    query_image = Image.open(img_path).convert("RGB")

    # 3. Trích xuất vector và Tìm kiếm
    with torch.no_grad():
        query_vec = model.encode_images([query_image]).cpu().numpy()
    
    # Tìm top 6 (vì top 1 chắc chắn là chính nó, ta xem 5 ảnh còn lại)
    top_k = 6
    scores, indices = vector_store.search(query_vec, top_k=top_k)

    # 4. Trực quan hóa kết quả
    fig, axes = plt.subplots(1, top_k, figsize=(25, 5))
    
    # Hiển thị ảnh Query (là vị trí số 1)
    axes[0].imshow(query_image)
    axes[0].set_title(f"QUERY (Index: {indices[0]})\nScore: {scores[0]:.2f}", color='red', fontweight='bold')
    axes[0].axis('off')
    
    # Hiển thị 5 ảnh tương đồng tiếp theo
    print("\n--- Top Similar Images Found ---")
    for i in range(1, top_k):
        idx = int(indices[i])
        score = scores[i]
        res_img = dataset[idx]['image']
        caption = dataset[idx]['caption'][0]
        
        axes[i].imshow(res_img)
        axes[i].set_title(f"Rank {i}\nScore: {score:.4f}")
        axes[i].axis('off')
        print(f"Rank {i}: Index {idx} | Score {score:.4f} | Caption: {caption[:60]}...")

    plt.suptitle(f"Image-to-Image Search Result for {random_file}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_random_demo()