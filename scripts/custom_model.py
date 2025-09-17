# Text-to-Image Search System tự xây dựng model
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from collections import Counter

# Vietnamese text processing
import string
import unicodedata

class VietnameseTextProcessor:
    """
    Xử lý text tiếng Việt
    """
    def __init__(self):
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def normalize_text(self, text):
        """
        Chuẩn hóa text tiếng Việt
        """
        # Chuyển thành lowercase
        text = text.lower()
        
        # Xóa dấu câu
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Xóa khoảng trắng thừa
        text = ' '.join(text.split())
        
        return text
    
    def build_vocab(self, texts, min_freq=1):
        """
        Xây dựng vocabulary từ danh sách texts
        """
        word_counts = Counter()
        
        for text in texts:
            normalized = self.normalize_text(text)
            words = normalized.split()
            word_counts.update(words)
        
        # Tạo vocab với min frequency
        vocab_words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # Thêm special tokens
        self.word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        for word in vocab_words:
            self.word2idx[word] = len(self.word2idx)
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"Built vocabulary with {self.vocab_size} words")
    
    def text_to_sequence(self, text, max_length=50):
        """
        Chuyển text thành sequence of indices
        """
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        # Chuyển words thành indices
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Padding hoặc truncate
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.word2idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def save_vocab(self, path):
        """
        Lưu vocabulary
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path):
        """
        Load vocabulary
        """
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word'] 
        self.vocab_size = vocab_data['vocab_size']
        print(f"Vocabulary loaded: {self.vocab_size} words")

class TourismDataset(Dataset):
    """
    Dataset cho tourism images và text
    """
    def __init__(self, metadata_file, text_processor, transform=None, max_text_length=50):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.text_processor = text_processor
        self.transform = transform
        self.max_text_length = max_text_length
        
        # Tạo positive và negative pairs
        self.create_pairs()
    
    def create_pairs(self):
        """
        Tạo positive/negative pairs cho contrastive learning
        """
        self.pairs = []
        
        for i, item in enumerate(self.metadata):
            # Positive pairs: (image, matching_text)
            texts = []
            
            # Thêm captions
            if 'captions' in item:
                if 'vi' in item['captions']:
                    texts.append(item['captions']['vi'])
                if 'en' in item['captions']:
                    texts.append(item['captions']['en'])
            
            # Thêm location info
            if 'location' in item:
                texts.append(item['location'].get('city', ''))
            
            # Thêm landmark
            if 'landmark' in item:
                texts.append(item['landmark'])
            
            # Thêm tags
            if 'tags' in item:
                texts.extend(item['tags'])
            
            # Tạo positive pairs
            for text in texts:
                if text and text.strip():
                    self.pairs.append({
                        'image_idx': i,
                        'text': text.strip(),
                        'label': 1  # Positive
                    })
            
            # Tạo negative pairs (random sampling)
            num_negatives = min(len(texts), 3)  # Tối đa 3 negative pairs per image
            negative_indices = np.random.choice(
                [j for j in range(len(self.metadata)) if j != i],
                size=num_negatives,
                replace=False
            )
            
            for neg_idx in negative_indices:
                neg_item = self.metadata[neg_idx]
                neg_text = ""
                
                # Lấy text từ negative item
                if 'captions' in neg_item and 'vi' in neg_item['captions']:
                    neg_text = neg_item['captions']['vi']
                elif 'location' in neg_item:
                    neg_text = neg_item['location'].get('city', '')
                
                if neg_text and neg_text.strip():
                    self.pairs.append({
                        'image_idx': i,
                        'text': neg_text.strip(),
                        'label': 0  # Negative
                    })
        
        print(f"Created {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load image
        image_item = self.metadata[pair['image_idx']]
        try:
            image = Image.open(image_item['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_item['image_path']}: {e}")
            # Tạo dummy image
            image = torch.zeros(3, 224, 224)
        
        # Process text
        text_sequence = self.text_processor.text_to_sequence(
            pair['text'], 
            self.max_text_length
        )
        text_tensor = torch.tensor(text_sequence, dtype=torch.long)
        
        return {
            'image': image,
            'text': text_tensor,
            'label': torch.tensor(pair['label'], dtype=torch.float),
            'text_str': pair['text']  # For debugging
        }

class TextEncoder(nn.Module):
    """
    Text encoder using LSTM
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, output_dim=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state (concatenate forward and backward)
        # hidden shape: (num_layers*2, batch_size, hidden_dim)
        hidden_forward = hidden[-2]  # Last layer forward
        hidden_backward = hidden[-1]  # Last layer backward
        
        combined = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # Project to output dimension
        output = self.fc(self.dropout(combined))
        
        # L2 normalize
        output = nn.functional.normalize(output, p=2, dim=1)
        
        return output

class ImageEncoder(nn.Module):
    """
    Image encoder using ResNet
    """
    def __init__(self, output_dim=512):
        super().__init__()
        
        # Load pretrained ResNet
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add projection head
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # (batch_size, 2048, 1, 1)
        
        # Project to output dimension
        output = self.projection(features)
        
        # L2 normalize
        output = nn.functional.normalize(output, p=2, dim=1)
        
        return output

class TextImageModel(nn.Module):
    """
    Combined text-image model
    """
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        
        self.text_encoder = TextEncoder(vocab_size, output_dim=embed_dim)
        self.image_encoder = ImageEncoder(output_dim=embed_dim)
        
    def forward(self, images, texts):
        text_features = self.text_encoder(texts)
        image_features = self.image_encoder(images)
        
        return text_features, image_features
    
    def encode_text(self, texts):
        return self.text_encoder(texts)
    
    def encode_image(self, images):
        return self.image_encoder(images)

def contrastive_loss(text_features, image_features, labels, temperature=0.07):
    """
    Contrastive loss function
    """
    # Tính similarity matrix
    similarity = torch.matmul(text_features, image_features.T) / temperature
    
    # Binary cross entropy loss cho từng pair
    batch_size = text_features.size(0)
    loss = 0
    
    for i in range(batch_size):
        # Positive/negative based on label
        if labels[i] == 1:  # Positive pair
            target = torch.zeros(batch_size).to(labels.device)
            target[i] = 1
            loss += nn.functional.cross_entropy(similarity[i].unsqueeze(0), torch.tensor([i]).to(labels.device))
        else:  # Negative pair - should have low similarity
            loss += torch.max(torch.tensor(0.0).to(labels.device), 
                            0.2 + similarity[i][i])  # Margin of 0.2
    
    return loss / batch_size

class CustomImageSearchSystem:
    """
    Hệ thống search tự xây dựng
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.text_processor = VietnameseTextProcessor()
        self.model = None
        self.image_embeddings = None
        self.metadata = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_data(self, metadata_file, vocab_path=None):
        """
        Chuẩn bị dữ liệu training
        """
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Collect all texts
        all_texts = []
        for item in metadata:
            if 'captions' in item:
                if 'vi' in item['captions']:
                    all_texts.append(item['captions']['vi'])
                if 'en' in item['captions']:
                    all_texts.append(item['captions']['en'])
            
            if 'location' in item and 'city' in item['location']:
                all_texts.append(item['location']['city'])
            
            if 'landmark' in item:
                all_texts.append(item['landmark'])
            
            if 'tags' in item:
                all_texts.extend(item['tags'])
        
        # Build vocabulary
        if vocab_path and os.path.exists(vocab_path):
            self.text_processor.load_vocab(vocab_path)
        else:
            self.text_processor.build_vocab(all_texts, min_freq=1)
            if vocab_path:
                self.text_processor.save_vocab(vocab_path)
        
        # Initialize model
        self.model = TextImageModel(self.text_processor.vocab_size).to(self.device)
        
        print(f"Model initialized with vocab size: {self.text_processor.vocab_size}")
    
     def train(self, metadata_file, num_epochs=50, batch_size=16, learning_rate=0.001, save_path="custom_model.pth"):
        """
        Train model
        """
        # Data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Dataset và DataLoader
        dataset = TourismDataset(metadata_file, self.text_processor, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                text_features, image_features = self.model(images, texts)
                
                # Compute loss
                loss = contrastive_loss(text_features, image_features, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item():.4f})
            
            # Cập nhật learning rate
            scheduler.step()
            
            # Tính average loss cho epoch
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint mỗi 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                self.save_model(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save final model
        self.save_model(save_path)
        print(f"Training completed! Final model saved: {save_path}")
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.show()
        
        return train_losses
    
    def save_model(self, save_path):
        """
        Lưu model và text processor
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.text_processor.vocab_size,
            'word2idx': self.text_processor.word2idx,
            'idx2word': self.text_processor.idx2word
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """
        Load model đã train
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Khôi phục text processor
        self.text_processor.vocab_size = checkpoint['vocab_size']
        self.text_processor.word2idx = checkpoint['word2idx']
        self.text_processor.idx2word = checkpoint['idx2word']
        
        # Khôi phục model
        self.model = TextImageModel(self.text_processor.vocab_size).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def encode_dataset_images(self, metadata_file, save_embeddings_path=None):
        """
        Encode tất cả images trong dataset
        """
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.image_embeddings = []
        self.model.eval()
        
        print("Encoding images...")
        with torch.no_grad():
            for idx, item in enumerate(tqdm(self.metadata)):
                try:
                    # Load và preprocess image
                    image = Image.open(item['image_path']).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Encode image
                    image_embedding = self.model.encode_image(image_tensor)
                    self.image_embeddings.append(image_embedding.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error processing {item.get('image_path', 'unknown')}: {e}")
                    # Tạo zero embedding cho lỗi
                    zero_embedding = torch.zeros(1, 512).numpy()
                    self.image_embeddings.append(zero_embedding)
        
        self.image_embeddings = np.vstack(self.image_embeddings)
        print(f"Encoded {len(self.image_embeddings)} images")
        
        # Lưu embeddings
        if save_embeddings_path:
            self.save_embeddings(save_embeddings_path)
    
    def save_embeddings(self, save_path):
        """
        Lưu embeddings và metadata
        """
        data = {
            'embeddings': self.image_embeddings,
            'metadata': self.metadata
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path):
        """
        Load embeddings đã tính sẵn
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.image_embeddings = data['embeddings']
        self.metadata = data['metadata']
        print(f"Loaded embeddings for {len(self.image_embeddings)} images")
    
    def search(self, text_query, top_k=5):
        """
        Tìm kiếm ảnh dựa trên text query
        """
        if self.image_embeddings is None:
            raise ValueError("No image embeddings loaded. Please run encode_dataset_images() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Encode text query
            text_sequence = self.text_processor.text_to_sequence(text_query)
            text_tensor = torch.tensor([text_sequence], dtype=torch.long).to(self.device)
            text_embedding = self.model.encode_text(text_tensor)
            text_embedding = text_embedding.cpu().numpy()
            
            # Tính cosine similarity
            similarities = cosine_similarity(text_embedding, self.image_embeddings)[0]
            
            # Lấy top k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                result = {
                    'metadata': self.metadata[idx],
                    'similarity_score': float(similarities[idx]),
                    'image_path': self.metadata[idx]['image_path']
                }
                results.append(result)
            
            return results
    
    def evaluate(self, test_metadata_file, top_k_list=[1, 5, 10]):
        """
        Đánh giá model với Recall@K
        """
        with open(test_metadata_file, 'r', encoding='utf-8') as f:
            test_metadata = json.load(f)
        
        recalls = {k: [] for k in top_k_list}
        
        for item in tqdm(test_metadata, desc="Evaluating"):
            # Tạo query từ caption hoặc location
            query = ""
            if 'captions' in item and 'vi' in item['captions']:
                query = item['captions']['vi']
            elif 'location' in item and 'city' in item['location']:
                query = item['location']['city']
            
            if not query:
                continue
            
            # Search
            results = self.search(query, top_k=max(top_k_list))
            
            # Tính recall cho từng k
            target_id = item['image_id']
            result_ids = [r['metadata']['image_id'] for r in results]
            
            for k in top_k_list:
                is_found = target_id in result_ids[:k]
                recalls[k].append(1.0 if is_found else 0.0)
        
        # Tính average recalls
        avg_recalls = {}
        for k in top_k_list:
            if recalls[k]:
                avg_recalls[f'Recall@{k}'] = np.mean(recalls[k])
            else:
                avg_recalls[f'Recall@{k}'] = 0.0
        
        print("Evaluation Results:")
        for metric, value in avg_recalls.items():
            print(f"{metric}: {value:.4f}")
        
        return avg_recalls

# Demo và utility functions
def create_extended_sample_dataset():
    """
    Tạo extended sample dataset cho training
    """
    locations = ['Đà Nẵng', 'Hà Nội', 'TP.HCM', 'Hội An', 'Sapa']
    categories = ['beach', 'mountain', 'architecture', 'street', 'landmark']
    
    sample_data = []
    
    for i, location in enumerate(locations):
        for j in range(5):  # 5 ảnh per location
            image_id = f"{location.lower().replace(' ', '_')}_{j+1:03d}"
            
            sample_item = {
                "image_id": image_id,
                "image_path": f"images/{location.lower().replace(' ', '_')}/{image_id}.jpg",
                "location": {
                    "city": location,
                    "country": "Việt Nam",
                    "region": "Vietnam"
                },
                "landmark": f"Landmark {j+1} in {location}",
                "captions": {
                    "vi": f"Hình ảnh đẹp của {location} vào {['sáng', 'trưa', 'chiều', 'tối', 'đêm'][j]}",
                    "en": f"Beautiful image of {location} in the {['morning', 'noon', 'afternoon', 'evening', 'night'][j]}"
                },
                "tags": [location.lower(), location.lower().replace(' ', ''), categories[i]],
                "category": categories[i],
                "time_taken": ['morning', 'noon', 'afternoon', 'evening', 'night'][j]
            }
            sample_data.append(sample_item)
    
    with open('extended_dataset_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"Extended sample dataset created: {len(sample_data)} items")
    return sample_data

# Main demo
if __name__ == "__main__":
    # Tạo extended dataset
    create_extended_sample_dataset()
    
    # Khởi tạo system
    search_system = CustomImageSearchSystem()
    
    # Chuẩn bị dữ liệu
    search_system.prepare_data('extended_dataset_metadata.json', 'vocab.pkl')
    
    # Training (comment out nếu đã có model)
    print("Starting training...")
    train_losses = search_system.train(
        'extended_dataset_metadata.json',
        num_epochs=20,
        batch_size=8,
        learning_rate=0.001,
        save_path='custom_tourism_model.pth'
    )
    
    # Encode dataset images
    search_system.encode_dataset_images(
        'extended_dataset_metadata.json',
        'custom_embeddings.pkl'
    )
    
    # Test search
    print("\n=== CUSTOM MODEL SEARCH RESULTS ===")
    test_queries = ["Đà Nẵng", "Hà Nội", "beach", "mountain"]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = search_system.search(query, top_k=3)
        
        for idx, result in enumerate(results):
            print(f"  {idx+1}. {result['image_path']} (score: {result['similarity_score']:.4f})")
            print(f"     Location: {result['metadata']['location']['city']}")
            print(f"     Caption: {result['metadata']['captions']['vi']}")
    
    # Evaluation
    print("\n=== EVALUATION ===")
    eval_results = search_system.evaluate('extended_dataset_metadata.json')
    
    print("\nTraining completed successfully!")
    print("Files created:")
    print("- custom_tourism_model.pth: Trained model")
    print("- vocab.pkl: Text vocabulary")  
    print("- custom_embeddings.pkl: Image embeddings")
    print("- training_loss.png: Training curve")
                