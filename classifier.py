import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.optim.lr_scheduler import ExponentialLR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class RAGClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        # Tokenizer와 모델 직접 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Embedding 모델 freeze 여부 선택 (주석 해제하면 freeze)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False  
        
        embedding_dim = self.encoder.config.hidden_size
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(embedding_dim, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 2)
        # )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )
    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 문장 임베딩 생성"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, texts):
        # Tokenize
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt',
            max_length=512
        )
        
        # GPU로 이동 
        if next(self.parameters()).is_cuda:
            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        
        # Encoding
        model_output = self.encoder(**encoded_input)
        
        # Mean pooling으로 문장 임베딩 생성
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Classification
        logits = self.classifier(embeddings)
        return logits

model = RAGClassifier('Qwen/Qwen3-Embedding-0.6B')

from torch.utils.data import DataLoader, Dataset

class QueryDataset(Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

# Prepare dataset examples 
train_queries = [
    "안녕하세요",              # 0: RAG 불필요
    "날씨 어때?",              # 0
    "파이썬 리스트는?",         # 0
    "우리 회사 매출은?",        # 1: RAG 필요
    "계약서 5조 내용은?",       # 1
    "최근 회의록 보여줘",       # 1
]
train_labels = [0, 0, 0, 1, 1, 1]

dataset = QueryDataset(train_queries, train_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

scheduler = ExponentialLR(optimizer,gamma=0.995)
criterion = nn.CrossEntropyLoss()

model.train()
epoch = 10
for epoch in range(epoch):
    for batch_texts, batch_labels in dataloader:
        optimizer.zero_grad()
        
        logits = model(batch_texts)
        loss = criterion(logits, torch.tensor(batch_labels))
        
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
torch.save(model, "./save_model/rag_classifier_full.pth")

# inference
model.eval()
with torch.no_grad():
    test_query = ["한국은 어디에 있어?"]
    logits = model(test_query)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    
    print(f"RAG 필요: {bool(pred)}, 확신도: {confidence:.2%}")
