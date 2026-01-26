import streamlit as st
import yaml
import torch
import os
from PIL import Image
from src.models.clip_wrapper import CLIPEngine
from src.models.siglip_wrapper import SigLIPEngine
from src.engine.faiss_index import FAISSIndex
from src.loaders.hf_dataset import load_flickr30k

# 1. CẤU HÌNH TRANG
st.set_page_config(layout="wide", page_title="Multimodal Retrieval System")

@st.cache_resource
def load_resources(choice):
    if "SigLIP" in choice:
        config_path = "configs/model/siglip.yaml"
        engine_class = SigLIPEngine
        index_file = "data/embeddings/siglip_flickr30k.index"
    else:
        config_path = "configs/model/clip.yaml"
        engine_class = CLIPEngine
        index_file = "data/embeddings/clip_flickr30k.index"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = engine_class(model_id=cfg['model']['model_id'], device=device)
    vector_store = FAISSIndex(dimension=cfg['model']['embedding_dim'])
    
    if os.path.exists(index_file):
        vector_store.load(index_file)
    else:
        st.warning(f"Chưa có tệp Index: {index_file}")

    dataset = load_flickr30k()
    return model, vector_store, dataset, cfg

# --- SIDEBAR ---
st.sidebar.title("🛠 Tùy chỉnh hệ thống")
model_choice = st.sidebar.selectbox(
    "Chọn phiên bản mô hình:",
    ["SigLIP (Base-P16)", "CLIP (ViT-B/32)"]
)
top_k = st.sidebar.slider("Số lượng ảnh hiển thị", 1, 15, 6)

model, vector_store, dataset, config = load_resources(model_choice)

# --- GIAO DIỆN CHÍNH ---
st.title(f"🔍 AI Multimodal Search: {model_choice}")
st.info(f"Không gian vector: **{config['model']['embedding_dim']} chiều**")

# Sử dụng Tabs để phân loại chế độ tìm kiếm
tab1, tab2 = st.tabs(["🖼️ Image-to-Image", "📝 Text-to-Image"])

# --- TAB 1: IMAGE SEARCH ---
with tab1:
    uploaded_file = st.file_uploader("Tải ảnh lên để tìm ảnh tương tự", type=['jpg', 'jpeg', 'png'], key="img_search")
    
    if uploaded_file:
        query_img = Image.open(uploaded_file).convert("RGB")
        col_q, col_r = st.columns([1, 2.5])
        
        with col_q:
            st.image(query_img, caption="Ảnh truy vấn", use_container_width=True)

        with col_r:
            with st.spinner("Đang tìm kiếm..."):
                with torch.no_grad():
                    query_vec = model.encode_images([query_img]).cpu().numpy()
                scores, indices = vector_store.search(query_vec, top_k=top_k + 1)
                
                # Logic hiển thị kết quả
                start_idx = 1 if scores[0] > 0.999 else 0
                final_indices = indices[start_idx : start_idx + top_k]
                final_scores = scores[start_idx : start_idx + top_k]
                
                # Hiển thị Grid
                cols = st.columns(3)
                for i, (idx, score) in enumerate(zip(final_indices, final_scores)):
                    with cols[i % 3]:
                        item = dataset[int(idx)]
                        st.image(item['image'], use_container_width=True)
                        st.metric("Similarity", f"{score:.4f}")
                        with st.expander("Caption"):
                            st.write(item['caption'][0])

# --- TAB 2: TEXT SEARCH ---
with tab2:
    text_query = st.text_input("Nhập mô tả hình ảnh bạn muốn tìm:", placeholder="Ví dụ: A dog playing in the park...")
    
    if text_query:
        with st.spinner(f"Đang tìm ảnh phù hợp với: '{text_query}'..."):
            # 1. Encode văn bản thành vector
            with torch.no_grad():
                query_vec = model.encode_text([text_query]).cpu().numpy()
            
            # 2. Tìm kiếm trong cùng tệp Index ảnh
            scores, indices = vector_store.search(query_vec, top_k=top_k)
            
            # 3. Hiển thị Grid kết quả
            cols = st.columns(3)
            for i, (idx, score) in enumerate(zip(indices, scores)):
                with cols[i % 3]:
                    item = dataset[int(idx)]
                    st.image(item['image'], use_container_width=True)
                    st.metric("Similarity", f"{score:.4f}")
                    with st.expander("Caption"):
                        st.write(item['caption'][0])