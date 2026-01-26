import torch
from src.models.clip_wrapper import CLIPEngine
from src.engine.faiss_index import FAISSIndex
from src.loaders.hf_dataset import load_flickr30k

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPEngine(device=device)
    vector_store = FAISSIndex(dimension=512)
    vector_store.load()
    dataset = load_flickr30k()

    print("\n[READY] System is live. Type your query below.")
    while True:
        query = input("\nSearch (or 'q' to quit): ")
        if query.lower() == 'q': break
        
        # Text-to-Image Search
        text_vec = model.encode_text([query]).cpu().numpy()
        scores, indices = vector_store.search(text_vec, top_k=5)
        
        print(f"Top results for '{query}':")
        for i, idx in enumerate(indices):
            print(f"{i+1}. Index: {idx} | Caption: {dataset[int(idx)]['caption'][0][:80]}...")

if __name__ == "__main__":
    main()