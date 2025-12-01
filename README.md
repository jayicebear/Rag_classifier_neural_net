# RAG 필요 여부 분류기  
### Qwen3-Embedding-0.6B + Custom Neural Network Classifier

## 한 줄 요약

**임베딩은 Qwen3-Embedding-0.6B**가 만들고,  
**RAG 필요 여부(0 or 1)는 직접 설계한 3-layer MLP가 판단**하는 경량 이진 분류기입니다.
  
(백본은 그대로 두고 classifier만 학습해도 되고, 필요하면 전체 fine-tuning도 가능)

## 모델 아키텍처

```text
[Input Text]
     ↓
Tokenizer → Qwen3-Embedding-0.6B (768-dim)
     ↓
Mean Pooling (with attention_mask)
     ↓
LayerNorm(768)
     ↓
Linear(768 → 512) → SiLU → Dropout(0.2)
     ↓
Linear(512 → 128) → SiLU
     ↓
Linear(128 → 2)
     ↓
[0: RAG 불필요 / 1: RAG 필요]```

<img src="https://img.shields.io/badge/Backbone-Qwen3--Embedding--0.6B-blue" alt="Backbone">
<img src="https://img.shields.io/badge/Classifier-Custom%20MLP-brightgreen" alt="Classifier">
<img src="https://img.shields.io/badge/Task-RAG%20Routing%20%28이진%20분류%29-orange" alt="Task">
<img src="https://img.shields.io/badge/Language-한국어%20·%20다국어-FFD21E" alt="Lang">
