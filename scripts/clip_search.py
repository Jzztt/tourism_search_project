import json
import os
import numpy as np
import torch
import clip
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

class CLIPImageSearch:
    def __init__(self, model_name="ViT-B/32"):
        """
        Khởi tạo CLIP model
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Lưu trữ embeddings
        self.image_embeddings = []
        self.image_metadata = []
        
    def load_dataset(self, metadata_path):
        """
        Load dataset từ file JSON metadata
        
        Args:
            metadata_path: Đường dẫn file JSON chứa metadata ảnh
        """
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} images")
        
    def encode_images(self, save_embeddings_path=None):
        """
        Encode tất cả images trong dataset thành embeddings
        
        Args:
            save_embeddings_path: Đường dẫn lưu embeddings (optional)
        """
        print("Encoding images...")
        self.image_embeddings = []
        self.image_metadata = []
        
        for idx, item in enumerate(self.dataset):
            try:
                # Load image
                image_path = item['image_path']
                image = Image.open(image_path).convert('RGB')
                
                # Preprocess và encode
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_embedding = self.model.encode_image(image_input)
                    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                
                self.image_embeddings.append(image_embedding.cpu().numpy())
                self.image_metadata.append(item)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(self.dataset)} images")
                    
            except Exception as e:
                print(f"Error processing {item.get('image_path', 'unknown')}: {e}")
                continue
        
        self.image_embeddings = np.vstack(self.image_embeddings)
        print(f"Successfully encoded {len(self.image_embeddings)} images")
        
        # Lưu embeddings nếu cần
        if save_embeddings_path:
            self.save_embeddings(save_embeddings_path)
    
    def save_embeddings(self, save_path):
        """
        Lưu embeddings và metadata
        """
        data = {
            'embeddings': self.image_embeddings,
            'metadata': self.image_metadata
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path):
        """
        Load embeddings đã được tính sẵn
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.image_embeddings = data['embeddings']
        self.image_metadata = data['metadata']
        print(f"Loaded embeddings for {len(self.image_embeddings)} images")
    
    def search(self, text_query, top_k=5):
        """
        Tìm kiếm ảnh dựa trên text query
        
        Args:
            text_query: Câu truy vấn text
            top_k: Số lượng ảnh trả về
            
        Returns:
            List các ảnh có điểm số cao nhất
        """
        if len(self.image_embeddings) == 0:
            raise ValueError("No image embeddings loaded. Please run encode_images() first.")
        
        # Encode text query
        text_tokens = clip.tokenize([text_query]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        text_embedding = text_embedding.cpu().numpy()
        
        # Tính cosine similarity
        similarities = cosine_similarity(text_embedding, self.image_embeddings)[0]
        
        # Lấy top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = {
                'metadata': self.image_metadata[idx],
                'similarity_score': float(similarities[idx]),
                'image_path': self.image_metadata[idx]['image_path']
            }
            results.append(result)
        
        return results
    
    def advanced_search(self, text_query, location_filter=None, category_filter=None, top_k=5):
        """
        Tìm kiếm nâng cao với filter
        
        Args:
            text_query: Câu truy vấn text
            location_filter: Filter theo thành phố
            category_filter: Filter theo category
            top_k: Số lượng ảnh trả về
        """
        # Lọc metadata trước
        filtered_indices = []
        for idx, metadata in enumerate(self.image_metadata):
            # Filter theo location
            if location_filter and metadata.get('location', {}).get('city', '').lower() != location_filter.lower():
                continue
            
            # Filter theo category  
            if category_filter and metadata.get('category', '').lower() != category_filter.lower():
                continue
            
            filtered_indices.append(idx)
        
        if not filtered_indices:
            return []
        
        # Encode text query
        text_tokens = clip.tokenize([text_query]).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        text_embedding = text_embedding.cpu().numpy()
        
        # Tính similarity chỉ cho filtered images
        filtered_embeddings = self.image_embeddings[filtered_indices]
        similarities = cosine_similarity(text_embedding, filtered_embeddings)[0]
        
        # Lấy top k results
        top_local_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for local_idx in top_local_indices:
            global_idx = filtered_indices[local_idx]
            result = {
                'metadata': self.image_metadata[global_idx],
                'similarity_score': float(similarities[local_idx]),
                'image_path': self.image_metadata[global_idx]['image_path']
            }
            results.append(result)
        
        return results

# Utility functions
def create_sample_dataset():
    """
    Tạo sample dataset để test
    """
    sample_data = [
        {
            "image_id": "danang_001",
            "image_path": "images/danang/danang_001.jpg",
            "location": {
                "city": "Đà Nẵng",
                "country": "Việt Nam",
                "region": "Miền Trung"
            },
            "landmark": "Cầu Rồng",
            "captions": {
                "vi": "Cầu Rồng Đà Nẵng lung linh về đêm",
                "en": "Dragon Bridge in Da Nang illuminated at night"
            },
            "tags": ["đà nẵng", "da nang", "cầu rồng", "dragon bridge", "landmark"],
            "category": "architecture",
            "time_taken": "night"
        },
        {
            "image_id": "danang_002", 
            "image_path": "images/danang/danang_002.jpg",
            "location": {
                "city": "Đà Nẵng",
                "country": "Việt Nam",
                "region": "Miền Trung"
            },
            "landmark": "Bãi biển Mỹ Khê",
            "captions": {
                "vi": "Bãi biển Mỹ Khê Đà Nẵng cát trắng nước trong",
                "en": "My Khe Beach in Da Nang with white sand and clear water"
            },
            "tags": ["đà nẵng", "da nang", "mỹ khê", "my khe beach", "beach"],
            "category": "beach", 
            "time_taken": "day"
        }
    ]
    
    with open('dataset_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("Sample dataset created: dataset_metadata.json")


# Demo usage
if __name__ == "__main__":
    # Khởi tạo search system
    search_system = CLIPImageSearch()
    
    # Tạo sample dataset (hoặc load dataset có sẵn)
    search_system.load_dataset("data/dataset_metadata.json")
    
    
    # Encode images (chỉ chạy 1 lần)
    search_system.encode_images("models/image_embeddings.pkl")
    
    # Hoặc load embeddings đã có
    # search_system.load_embeddings('image_embeddings.pkl')
    
    # Test search
    print("\n=== SEARCH RESULTS ===")
    results = search_system.search(" Biển Đà Nẵng", top_k=3)
    
    for idx, result in enumerate(results):
        print(f"\nResult {idx+1}:")
        print(f"Image: {result['image_path']}")
        print(f"Score: {result['similarity_score']:.4f}")
        print(f"Location: {result['metadata']['location']['city']}")
        print(f"Caption: {result['metadata']['captions']['vi']}")
    
    # Test advanced search
    print("\n=== ADVANCED SEARCH RESULTS ===")
    results = search_system.advanced_search(
        "beach", 
        location_filter="Đà Nẵng",
        top_k=2
    )
    
    for idx, result in enumerate(results):
        print(f"\nResult {idx+1}:")
        print(f"Image: {result['image_path']}")
        print(f"Score: {result['similarity_score']:.4f}")
        print(f"Category: {result['metadata']['category']}")