import os
from datasets import load_dataset
from tqdm import tqdm

def save_test_samples(num_samples=1000, save_dir="data/test_samples"):
    # 1. Tạo thư mục lưu trữ nếu chưa có
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"[INFO] Đã tạo thư mục: {save_dir}")

    # 2. Tải tập dữ liệu
    print("[INFO] Đang tải tập dữ liệu Flickr30k...")
    ds = load_dataset("nlphuji/flickr30k", split='test', trust_remote_code=True)

    # 3. Lặp và lưu ảnh
    print(f"[INFO] Đang lưu {num_samples} ảnh vào {save_dir}...")
    for i in tqdm(range(num_samples)):
        sample = ds[i]
        image = sample['image']
        
        # Đảm bảo ảnh ở hệ màu RGB để tránh lỗi khi lưu
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Lưu ảnh với tên: sample_0.jpg, sample_1.jpg, ...
        image_path = os.path.join(save_dir, f"sample_{i}.jpg")
        image.save(image_path)

    print(f"\n[SUCCESS] Đã lưu xong 1.000 ảnh. Bạn có thể tìm thấy chúng tại: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    save_test_samples()