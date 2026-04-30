# 02. Transformers for Machine Learning

---

## Why Transformers?
Transformers revolutionized Machine Learning in 2017 with the paper "Attention Is All You Need". They are the foundation of modern AI models like GPT, BERT, Claude, and ChatGPT. Unlike traditional neural networks, Transformers can process entire sequences in parallel and capture long-range dependencies efficiently.

---

## 📚 Core Concepts

### 1. **The Problem with Previous Models**

**Recurrent Neural Networks (RNNs):**
- Process sequences one token at a time (slow)
- Struggle with long-range dependencies
- Cannot parallelize training

**Transformers solve these problems by:**
- Processing entire sequences at once (parallel)
- Using attention to focus on relevant parts
- No sequential dependencies

---

### 2. **Attention Mechanism** 🎯

#### **What is Attention?**
Attention allows the model to focus on different parts of the input when processing each element. Think of it like reading a sentence - you pay more attention to certain words based on context.

**Example:**
```
Sentence: "The cat sat on the mat"
When processing "it", attention helps focus on "cat"
"The cat sat on the mat, and it was sleeping"
                          ↑
                    Attention to "cat"
```

#### **Mathematical Definition**

**Attention Formula:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Where:**
- $Q$ = Query matrix (what we're looking for)
- $K$ = Key matrix (what we're comparing against)  
- $V$ = Value matrix (the actual information)
- $d_k$ = dimension of keys (scaling factor)

**Why the formula works:**
1. $QK^T$ - Compute similarity (dot product) between query and all keys
2. $\frac{1}{\sqrt{d_k}}$ - Scale to prevent large values that could cause vanishing gradients
3. $\text{softmax}(\cdot)$ - Convert scores to probabilities (attention weights that sum to 1)
4. Multiply by $V$ - Weight the values by attention scores to get output

**Python Example:**
```python
import numpy as np

def softmax(x):
    """Compute softmax values"""
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    """
    Compute attention scores
    Q: Query matrix (seq_len, d_k)
    K: Key matrix (seq_len, d_k)
    V: Value matrix (seq_len, d_v)
    """
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores)
    
    # Multiply weights by values
    output = np.dot(attention_weights, V)
    
    return output, attention_weights

# Example usage
seq_len = 3
d_k = 4

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, weights = attention(Q, K, V)
print("Attention output shape:", output.shape)
print("Attention weights:\n", weights)
```

---

### 3. **Self-Attention**

**Self-Attention** means the sequence attends to itself. Each word looks at all other words (including itself) to understand context.

**Example:**
```
Sentence: "The animal didn't cross the street because it was too tired"

When processing "it":
- High attention to "animal" (it refers to the animal)
- Low attention to "street"
```

**Implementation:**
```python
def self_attention(X, W_q, W_k, W_v):
    """
    Self-attention where Q, K, V come from the same input
    X: Input embeddings (seq_len, d_model)
    W_q, W_k, W_v: Weight matrices for Q, K, V projections
    """
    # Project input to Q, K, V
    Q = np.dot(X, W_q)  # (seq_len, d_k)
    K = np.dot(X, W_k)  # (seq_len, d_k)
    V = np.dot(X, W_v)  # (seq_len, d_v)
    
    # Compute attention
    output, weights = attention(Q, K, V)
    return output, weights

# Example
seq_len = 4
d_model = 512  # Embedding dimension
d_k = 64       # Query/Key dimension

X = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_k)
W_k = np.random.randn(d_model, d_k)
W_v = np.random.randn(d_model, d_k)

output, weights = self_attention(X, W_q, W_k, W_v)
print("Self-attention output shape:", output.shape)
```

---

### 4. **Multi-Head Attention** 🎭

Instead of one attention mechanism, use multiple "heads" in parallel. Each head learns different relationships.

**Why Multiple Heads?**
- Head 1 might focus on syntax
- Head 2 might focus on semantics
- Head 3 might focus on long-range dependencies

**Formula:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O
$$

**Where each head is computed as:**

$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

**Parameters:**
- $W^Q_i, W^K_i, W^V_i$ = Projection matrices for head $i$
- $W^O$ = Output projection matrix
- $h$ = Number of heads

**Implementation:**
```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        d_model: Embedding dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        """
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
    
    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)
    
    def forward(self, X):
        """
        X: Input (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.shape
        
        # Linear projections
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Split into multiple heads
        Q = self.split_heads(Q.reshape(batch_size, seq_len, self.d_model))
        K = self.split_heads(K.reshape(batch_size, seq_len, self.d_model))
        V = self.split_heads(V.reshape(batch_size, seq_len, self.d_model))
        
        # Compute attention for each head
        # (simplified - in practice, use batch operations)
        
        # Concatenate heads and apply final linear layer
        # output = concat(heads) × W_o
        
        return output

# Example usage
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 1

mha = MultiHeadAttention(d_model, num_heads)
X = np.random.randn(batch_size, seq_len, d_model)
# output = mha.forward(X)
```

---

### 5. **Positional Encoding** 📍

Since Transformers process sequences in parallel, they need a way to know the position of each token.

**Why?**
- "Dog bites man" vs "Man bites dog" - order matters!
- Without position info, the model can't distinguish order

**Sinusoidal Positional Encoding:**

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

**Where:**
- $pos$ = position in the sequence (0, 1, 2, ...)
- $i$ = dimension index (0 to $d_{\text{model}}/2$)
- $d_{\text{model}}$ = embedding dimension
- Even dimensions use sine, odd dimensions use cosine

**Implementation:**
```python
def positional_encoding(seq_len, d_model):
    """
    Generate positional encodings
    seq_len: Length of sequence
    d_model: Embedding dimension
    """
    # Initialize position and dimension indices
    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]     # (1, d_model)
    
    # Calculate angle rates
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle = pos * angle_rates
    
    # Apply sin to even indices, cos to odd indices
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle[:, 1::2])
    
    return pos_encoding

# Example
seq_len = 50
d_model = 512
pos_enc = positional_encoding(seq_len, d_model)
print("Positional encoding shape:", pos_enc.shape)

# Visualization (would need matplotlib)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.imshow(pos_enc, cmap='RdBu', aspect='auto')
# plt.colorbar()
# plt.xlabel('Embedding dimension')
# plt.ylabel('Position')
# plt.title('Positional Encoding')
```

---

### 6. **Feed-Forward Network** 🔄

After attention, each position passes through a feed-forward network independently.

**Formula:**

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

**Where:**
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ = First weight matrix
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ = Second weight matrix
- $b_1 \in \mathbb{R}^{d_{ff}}$ = First bias vector
- $b_2 \in \mathbb{R}^{d_{\text{model}}}$ = Second bias vector
- $\text{ReLU}(x) = \max(0, x)$ = Rectified Linear Unit activation
- $d_{ff}$ = Feed-forward dimension (typically $4 \times d_{\text{model}}$)

**Implementation:**
```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        """
        d_model: Input/output dimension (e.g., 512)
        d_ff: Hidden layer dimension (e.g., 2048)
        """
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def forward(self, x):
        """
        x: Input (batch_size, seq_len, d_model)
        """
        # First linear transformation + ReLU
        hidden = self.relu(np.dot(x, self.W1) + self.b1)
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

# Example
d_model = 512
d_ff = 2048
seq_len = 10

ff = FeedForward(d_model, d_ff)
x = np.random.randn(1, seq_len, d_model)
output = ff.forward(x)
print("Feed-forward output shape:", output.shape)
```

---

### 7. **Layer Normalization** 📊

Normalize the inputs across features to stabilize training.

**Formula:**

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

**Where:**
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ = Mean across features
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$ = Variance across features
- $\gamma \in \mathbb{R}^d$ = Learnable scale parameter (initialized to 1)
- $\beta \in \mathbb{R}^d$ = Learnable shift parameter (initialized to 0)
- $\epsilon$ = Small constant for numerical stability (e.g., $10^{-6}$)
- $\odot$ = Element-wise multiplication
- $d$ = Feature dimension

**Implementation:**
```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        """
        d_model: Dimension to normalize over
        eps: Small constant for numerical stability
        """
        self.gamma = np.ones(d_model)  # Scale parameter
        self.beta = np.zeros(d_model)   # Shift parameter
        self.eps = eps
    
    def forward(self, x):
        """
        x: Input (batch_size, seq_len, d_model)
        """
        # Compute mean and variance across last dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_norm + self.beta
        
        return output

# Example
d_model = 512
ln = LayerNorm(d_model)
x = np.random.randn(1, 10, d_model)
output = ln.forward(x)
print("Layer norm output shape:", output.shape)
```

---

## 🏗 Transformer Architecture

### **Encoder Block**

Each encoder layer consists of:
1. Multi-Head Self-Attention
2. Add & Norm (residual connection + layer norm)
3. Feed-Forward Network
4. Add & Norm

```python
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """Single Transformer encoder layer"""
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x):
        """
        x: Input (batch_size, seq_len, d_model)
        """
        # Multi-head attention
        attn_output = self.mha.forward(x)
        
        # Add & Norm (residual connection)
        x = self.layernorm1.forward(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn.forward(x)
        
        # Add & Norm
        output = self.layernorm2.forward(x + ffn_output)
        
        return output
```

### **Decoder Block**

Each decoder layer has:
1. Masked Multi-Head Self-Attention (can't see future tokens)
2. Add & Norm
3. Multi-Head Cross-Attention (attends to encoder output)
4. Add & Norm
5. Feed-Forward Network
6. Add & Norm

```python
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """Single Transformer decoder layer"""
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.layernorm3 = LayerNorm(d_model)
    
    def forward(self, x, encoder_output, look_ahead_mask=None):
        """
        x: Decoder input
        encoder_output: Output from encoder
        look_ahead_mask: Mask to prevent looking at future tokens
        """
        # Masked self-attention
        attn1 = self.masked_mha.forward(x)  # Should use mask
        x = self.layernorm1.forward(x + attn1)
        
        # Cross-attention with encoder output
        attn2 = self.cross_mha.forward(x)  # Q from x, K,V from encoder_output
        x = self.layernorm2.forward(x + attn2)
        
        # Feed-forward
        ffn_output = self.ffn.forward(x)
        output = self.layernorm3.forward(x + ffn_output)
        
        return output
```

### **Complete Transformer**

```python
class Transformer:
    def __init__(self, 
                 vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 max_seq_len=5000,
                 dropout=0.1):
        """
        Complete Transformer model
        
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        """
        self.d_model = d_model
        
        # Embedding layers
        self.embedding = np.random.randn(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        
        # Encoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ]
        
        # Final linear layer
        self.final_layer = np.random.randn(d_model, vocab_size)
    
    def encode(self, input_tokens):
        """
        Encode input sequence
        input_tokens: (batch_size, seq_len)
        """
        # Embedding + positional encoding
        x = self.embedding[input_tokens]  # Lookup embeddings
        seq_len = input_tokens.shape[1]
        x += self.pos_encoding[:seq_len, :]
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer.forward(x)
        
        return x
    
    def decode(self, target_tokens, encoder_output):
        """
        Decode target sequence
        target_tokens: (batch_size, seq_len)
        encoder_output: Output from encoder
        """
        # Embedding + positional encoding
        x = self.embedding[target_tokens]
        seq_len = target_tokens.shape[1]
        x += self.pos_encoding[:seq_len, :]
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer.forward(x, encoder_output)
        
        # Final linear projection
        logits = np.dot(x, self.final_layer)
        
        return logits
    
    def forward(self, input_tokens, target_tokens):
        """
        Full forward pass
        """
        encoder_output = self.encode(input_tokens)
        decoder_output = self.decode(target_tokens, encoder_output)
        return decoder_output

# Example usage
vocab_size = 10000
transformer = Transformer(vocab_size)
print("Transformer created with:")
print(f"- {len(transformer.encoder_layers)} encoder layers")
print(f"- {len(transformer.decoder_layers)} decoder layers")
print(f"- {transformer.d_model} embedding dimensions")
```

---

## 🎯 Practical Applications

### 1. **Machine Translation**
```
Input: "Hello, how are you?" (English)
Output: "Bonjour, comment allez-vous?" (French)

Encoder processes English sentence
Decoder generates French translation
```

### 2. **Text Summarization**
```
Input: Long article (encoder)
Output: Short summary (decoder)
```

### 3. **Question Answering**
```
Input: Context + Question
Output: Answer extracted from context
```

### 4. **GPT Models (Decoder-only)**
```
GPT, ChatGPT, Claude use only the decoder part
Train on massive text to predict next token
```

### 5. **BERT Models (Encoder-only)**
```
BERT uses only the encoder part
Pre-train on masked language modeling
Fine-tune for classification, NER, etc.
```

---

## 📝 Key Differences: Transformer vs RNN/LSTM

| Feature | RNN/LSTM | Transformer |
|---------|----------|-------------|
| **Processing** | Sequential | Parallel |
| **Long-range deps** | Difficult | Easy (via attention) |
| **Training speed** | Slow | Fast |
| **Memory** | Limited | Full sequence access |
| **Parallelization** | No | Yes |

---

## 🔑 Key Takeaways

1. **Attention** lets models focus on relevant parts of input
2. **Self-Attention** allows tokens to attend to each other
3. **Multi-Head Attention** learns different relationships
4. **Positional Encoding** provides sequence order information
5. **Transformers** are the foundation of modern NLP
6. **Encoder-Decoder** structure for sequence-to-sequence tasks
7. **Residual connections** and **Layer Norm** stabilize training

---

## 📚 Practice Exercises

### Exercise 1: Implement Attention
```python
import numpy as np

# Implement the attention function
# Test with:
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0]])

# Expected: Attention should return weighted sum of V
```

### Exercise 2: Positional Encoding Visualization
```python
# Generate positional encodings for seq_len=100, d_model=128
# Plot them using matplotlib to visualize the pattern
```

### Exercise 3: Token-level Attention Weights
```python
# Given a sentence: "The cat sat on the mat"
# Simulate attention weights showing which words "it" attends to
sentence = ["The", "cat", "sat", "on", "the", "mat"]
# Create mock attention scores
```

### Exercise 4: Compare Attention Heads
```python
# Implement 2 attention heads
# Show how they might focus on different aspects
# Head 1: syntactic relationships
# Head 2: semantic relationships
```

---

## 🚀 Modern Transformer Variants

### **Decoder-Only Models**
- **GPT** (Generative Pre-trained Transformer)
- **GPT-2, GPT-3, GPT-4** (OpenAI)
- **Claude** (Anthropic)
- **LLaMA** (Meta)

### **Encoder-Only Models**
- **BERT** (Bidirectional Encoder Representations)
- **RoBERTa** (Robustly optimized BERT)
- **DistilBERT** (Smaller, faster BERT)

### **Encoder-Decoder Models**
- **T5** (Text-to-Text Transfer Transformer)
- **BART** (Bidirectional and Auto-Regressive Transformer)
- **Original Transformer** (machine translation)

---

## 🔗 Resources

- **Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Illustrated Transformer**: https://jalammar.github.io/illustrated-transformer/
- **The Annotated Transformer**: http://nlp.seas.harvard.edu/annotated-transformer/
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers/
- **3Blue1Brown**: Visual explanations of attention

---

## 💡 Next Steps

After mastering Transformers:
1. Implement a simple Transformer from scratch
2. Train on a small dataset (e.g., language modeling)
3. Explore pre-trained models (BERT, GPT)
4. Fine-tune for specific tasks
5. Learn about:
   - Tokenization (BPE, WordPiece)
   - Training techniques (AdamW, learning rate schedules)
   - Efficient Transformers (Flash Attention, etc.)

---

**Remember**: Transformers are complex, but understanding the core concepts (attention, multi-head attention, positional encoding) is key. Start with simple examples and gradually build up to complete implementations.

---

## 🎓 Advanced Topics (Optional)

### **Flash Attention**
- Optimized attention computation
- Reduces memory usage
- Speeds up training

### **Rotary Position Embeddings (RoPE)**
- Alternative to sinusoidal encoding
- Used in modern models (LLaMA, etc.)

### **Mixture of Experts (MoE)**
- Conditionally activate different parameters
- Scale models efficiently

### **Long Context Transformers**
- Handle sequences longer than training length
- Techniques: Sparse attention, hierarchical attention

---
